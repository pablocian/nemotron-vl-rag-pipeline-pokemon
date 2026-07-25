import os
import logging
import gradio as gr
import spaces
import torch
from pathlib import Path
from datasets import load_dataset

from src.pipeline import (
    Pipeline,
    Retriever,
    Generator,
)
from src.utils import (
    load_embed_model,
    load_embeddings,
    load_gen_model,
)

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))

TITLE_AND_DESCRIIPTION = """# 👁️📑 Multimodal RAG over Pokémon documents with Nemotron Embed VL

Chat with a [Pokémon database](https://huggingface.co/datasets/cian-ps/pokedb-vl-1k).
Input text and optionally an image and get a contextual response.

- **Embedding model used:** [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)
- **Generation model used:** [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
"""


logger.info("Loading data & models...")

ds = load_dataset("cian-ps/pokedb-vl-1k", split="train")

embeddings_path = PROJECT_ROOT / "image_text_embeddings.safetensors"
if not embeddings_path.exists():
    raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

embeddings = load_embeddings(embeddings_path)
embed_model = load_embed_model()
gen_model, gen_processor = load_gen_model()

rag_pipeline = Pipeline(
    retriever=Retriever(embed_model, ds, embeddings["image_text_embeddings"]),
    generator=Generator(gen_model, gen_processor),
)


@spaces.GPU
def on_query(query_text, query_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rag_pipeline.to(device)
    try:
        response, doc_images = rag_pipeline.cot_generation(query_text, query_image)
        return doc_images, response
    finally:
        rag_pipeline.to("cpu")


with gr.Blocks(title="Nemotron VL RAG - Pokémon") as app:
    gr.Markdown(TITLE_AND_DESCRIIPTION)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Query Input")

            query_text = gr.Textbox(
                label="Query Text",
                placeholder="e.g. 'How many evolutions does Eevee have?'",
                lines=2,
            )

            query_image = gr.Image(
                label="Query Image (optional)",
                type="pil",
                height=200,
            )

            enter_btn = gr.Button("Enter", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### Response:")

            response = gr.Markdown(
                "Hi! I'm an AI assistant designed to assist with questions related to Pokémon. How can I help you today?",
                label="Generated Response",
            )

            gallery_output = gr.Gallery(
                label="Retrieved Documents",
                columns=3,
                height="auto",
                object_fit="cover",
                show_label=True,
            )

    gr.Markdown("### Example Queries")

    gr.Examples(
        examples=[
            ["How many evolutions does Eevee have?"],
            ["Lucario is what dual type of Pokemon?"],
            ["What Pokedex number is Pikachu?"],
        ],
        inputs=[query_text],
        label="Example Queries",
    )

    enter_btn.click(
        fn=on_query,
        inputs=[query_text, query_image],
        outputs=[gallery_output, response],
    )


if __name__ == "__main__":
    logger.info("Launching app...")
    app.launch()
