from __future__ import annotations

from typing import Literal, Any, Dict
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoModel,
)
import torch
from safetensors.torch import load_file
from pathlib import Path


def load_embed_model(modality : Literal["image_text", "image", "text"] = "image_text") -> Any:
    max_tokens = {
        "image": 2048,
        "text": 8192,
        "image_text": 10240,
    }

    embed_model = AutoModel.from_pretrained(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval()

    # set max number of tokens
    embed_model.processor.p_max_length = max_tokens[modality]
    # Sets max number of tiles an image can be split. Each tile consumes 256 tokens.
    embed_model.processor.max_input_tiles = 6
    # Enables an extra tile with the full image at lower resolution
    embed_model.processor.use_thumbnail = True

    return embed_model


def load_embeddings(embeddings_path: str | Path) -> Dict[str, torch.Tensor]:
    return load_file(embeddings_path, device="cpu")


def load_gen_model() -> tuple[Qwen3VLForConditionalGeneration, Any]:
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
    )

    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    return qwen_model, qwen_processor
