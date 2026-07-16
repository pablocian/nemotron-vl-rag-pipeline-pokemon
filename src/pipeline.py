from __future__ import annotations

from typing import Any, Dict
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration
from datasets import Dataset


POKEMON_CHAT_TEMPLATE = """You are having a conversation with the user about Pokémon.
Use the documents below to formulate the best response you can to the user's question.
Documents are seperated by `\n\n---\n\n`. Only one or multiple documents may be relevant to the question.

<documents>
{docs}
</documents>

Question: {query}
Answer: """


class Pipeline:
    def __init__(self, retriever: Retriever, parser: Parser, generator: Generator) -> None:
        self.retriever = retriever
        self.parser = parser
        self.generator = generator

    def to(self, device: str | torch.device) -> Pipeline:
        self.retriever.to(device)
        self.generator.to(device)
        return self
    
    def run(self, query: str, query_image: Image.Image | None = None, docs_top_k: int = 3) -> tuple[str, list[Dict[str, Any]]]:
        """
        Multimodal RAG pipeline:

        Text (+ image) -> Retriever -> RAG template -> Language Model -> Response + top k retrieved document images

        Args:
            query(str): The user's query.
            query_image(PIL.Image.Image): The user's query image. Defaults to None.
            docs_top_k(int): The number of documents to retrieve. Defaults to 3.

        Returns:
            tuple[str, list[Dict[str, Any]]]: The response and the top k retrieved images and scores.
        """
        retrieved_docs, scores = self.retriever._run(query, query_image, docs_top_k)
        prompt, doc_images = self.parser._run(query, retrieved_docs)
        response = self.generator._run(prompt, query_image)
        
        # merge retrieved images and scores
        doc_images = [{"image": im, "score": score} for im, score in zip(doc_images, scores)]
        
        return response, doc_images


class Retriever:
    def __init__(self, embed_model: Any, ds: Dataset, ds_embeddings: torch.Tensor) -> None:
        self.embed_model = embed_model
        self.ds = ds
        self.ds_embeddings = ds_embeddings

    def to(self, device: str | torch.device) -> Retriever:
        self.embed_model.to(device)
        self.ds_embeddings = self.ds_embeddings.to(device)
        return self
    
    def _l2_normalize(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def _retrieve(
        self,
        embed_queries: torch.Tensor,
        embed_docs: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sim = self._l2_normalize(embed_queries) @ self._l2_normalize(embed_docs).T
        cos_sim_flat = cos_sim.flatten()
        top_k_idx = torch.argsort(cos_sim_flat, descending=True)[:top_k]
        top_k_scores = cos_sim_flat[top_k_idx]
        return top_k_idx, top_k_scores
    
    def _run(self, query: str, query_image: Image.Image | None, top_k: int) -> tuple[Dataset, list[float]]:
        with torch.inference_mode():
            if query_image is None:
                embed_query = self.embed_model.encode_documents(texts=[query])
            else:
                embed_query = self.embed_model.encode_documents(images=[query_image], texts=[query])

        indices, scores = self._retrieve(
            embed_query,
            self.ds_embeddings,
            top_k,
        )

        return self.ds.select(indices), scores.tolist()


class Parser:
    def __init__(self, prompt_template: str) -> None:
        self.prompt_template = prompt_template
    
    def _run(self, query: str, retrieved_docs: Dataset) -> tuple[str, list[Image.Image]]:
        docs = "\n\n---\n\n".join([doc["markdown_text"].rstrip() for doc in retrieved_docs])
        prompt = self.prompt_template.format(
            docs=docs,
            query=query,
        )
        return prompt, [doc["image"] for doc in retrieved_docs]


class Generator:
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

    def to(self, device: str | torch.device) -> Generator:
        self.gen_model.to(device)
        return self
    
    def _run(self, prompt: str, query_image: Image.Image | None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if query_image is not None:
            messages[0]["content"].append({"type": "image", "image": query_image})
        
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
