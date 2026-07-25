from __future__ import annotations

from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi
import torch
from PIL import Image
import json
from transformers import Qwen3VLForConditionalGeneration
from datasets import Dataset


REWRITE_OUTPUT_FORMAT_PROMPT = """Return your output as follows:

{"queries": ["<Variation 1>", "<Variation 2>", "<Variation 3>"]}

"""

REWRITE_PROMPT = """You are a helpful assistant that rephrases questions.

Given a question, generate three different variations of the question.

Variation 1: Same exact meaning. Remove conversational language and make sure the question is grammatically correct.
Variation 2: Ask a subquestion that needs to be answered to answer the original question.
Variation 3: Ask a more generic question that gets at the underlying principles needed to answer the original question.

If there are acronyms or words you are not familiar with, DO NOT try to rephrase them.

{output_format_prompt}

Question: {query}

Three variations of the question:
"""

COT_PROMPT = """Below you are given a list of documents as context and a list of questions.
Different document chunks are separated by three dashes (---).
Different questions are separated by three dashes, too.

Documents:
{documents}

Questions:
{questions}

For each question go through the following steps:
Step 1. What is the question's essential meaning and how does it relate to the documents?
Step 2. Which parts of the documents directly apply to the question?
Step 3. Can you answer the question based on evidence provided in the documents?
Step 4. Refine your answer with each question.

Provide your output as a final summary of your findings. Your output MUST NOT exceed 300 words.

Output:
"""

SIMPLE_QA_PROMPT = """
Generate a final concise and factual answer.
Your answer must be directly answering the following question in a conversational setting.

Question: {question}

Answer:
"""


class Pipeline:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def to(self, device: str | torch.device) -> Pipeline:
        self.retriever.to(device)
        self.generator.to(device)
        return self
    
    def query_rewriter(self, query: str) -> list[str]:
        output = self.generator.run_single_user_message(REWRITE_PROMPT.format(output_format_prompt=REWRITE_OUTPUT_FORMAT_PROMPT, query=query))
        try:
            queries = json.loads(output).get("queries")
            assert len(queries) == 3
        except:
            queries = [query]
        
        return queries

    def cot_generation(
        self, query: str, query_image: Image.Image | None = None, top_k: int = 10
    ) -> tuple[str, list[Image.Image]]:
        queries = self.query_rewriter(query)
        ranking = self.retriever.retrieve(queries, query_image)[:top_k]
        retrieved = self.retriever.ds.select(ranking)
        context = "\n\n---\n\n".join(retrieved["markdown_text"])
        doc_images = retrieved["image"]
        questions = "\n\n---\n\n".join(queries)

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are having a conversation with the users about different Pokémon species."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT.format(documents=context, questions=questions)},
                ],
            }
        ]

        if query_image is not None:
            messages[-1]["content"].append({"type": "image", "image": query_image})
        
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": self.generator.run_messages(messages)},
            ],
        })

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": SIMPLE_QA_PROMPT.format(question=queries[0])},
            ],
        })

        if query_image is not None:
            messages[-1]["content"].append({"type": "image", "image": query_image})
        
        return self.generator.run_messages(messages), doc_images


class Retriever:
    def __init__(self, embed_model: Any, ds: Dataset, ds_embeddings: torch.Tensor) -> None:
        self.embed_model = embed_model
        self.ds = ds
        self.ds_embeddings = ds_embeddings
        self.text_store = [doc.lower().split() for doc in ds["markdown_text"]]
        self.bm25 = BM25Okapi(self.text_store)

    def to(self, device: str | torch.device) -> Retriever:
        self.embed_model.to(device)
        self.ds_embeddings = self.ds_embeddings.to(device)
        return self

    def _l2_normalize(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def _semantic_search(
        self,
        embed_queries: torch.Tensor,
        embed_docs: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = self._l2_normalize(embed_queries) @ self._l2_normalize(embed_docs).T
        rankings = torch.argsort(cos_sim, dim=-1, descending=True)
        return rankings

    def _bm25_search(self, query: str) -> np.ndarray:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        return np.argsort(scores, descending=True)

    def _rrf_merge(self, r_semantic: torch.Tensor, r_lexical: np.ndarray | None = None) -> np.ndarray:
        k_constant = 60  # The value of 60 is a standard academic convention
        rankings = r_semantic.cpu().numpy()
        if r_lexical is not None:
            rankings = np.vstack((rankings, r_lexical))
        rrf_scores = np.zeros(rankings.shape[1])
        for r in rankings:
            for rank, idx in enumerate(r):
                rrf_scores[idx] += 1 / (k_constant + rank + 1)
        return np.argsort(rrf_scores, descending=True)

    def retrieve(self, queries: list[str], query_image: Image.Image | None = None, hybrid: bool = True) -> np.ndarray:
        with torch.inference_mode():
            if query_image is None:
                embed_queries = self.embed_model.encode_documents(texts=queries)
            else:
                embed_queries = self.embed_model.encode_documents(images=[query_image] * len(queries), texts=queries)
        
        r_semantic = self._semantic_search(embed_queries, self.ds_embeddings)
        r_lexical = None
        if hybrid:
            r_lexical = self._bm25_search(queries[0])
        
        final_ranking = self._rrf_merge(r_semantic, r_lexical)
        return final_ranking


class VLM:
    def __init__(
        self,
        gen_model: Qwen3VLForConditionalGeneration,
        gen_processor: Any,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ) -> None:
        self.gen_model = gen_model
        self.gen_processor = gen_processor
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, Any]]) -> str:
        inputs = self.gen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.gen_model.device)

        with torch.no_grad():
            output_ids = self.gen_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
            )
        
        output_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.gen_processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]


class Generator(VLM):
    def __init__(
        self,
        gen_model: Qwen3VLForConditionalGeneration,
        gen_processor: Any,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(gen_model, gen_processor, temperature, max_new_tokens)

    def to(self, device: str | torch.device) -> Generator:
        self.gen_model.to(device)
        return self
    
    def run_single_user_message(self, query: str, query_image: Image.Image | None = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            }
        ]

        if query_image is not None:
            messages[0]["content"].append({"type": "image", "image": query_image})
        
        return self.generate(messages)

    def run_messages(self, messages: list[dict[str, Any]]) -> str:
        return self.generate(messages)
