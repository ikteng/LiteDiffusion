"""Lazy-loaded diffusion pipelines and the core image generation call."""

from __future__ import annotations

import time

import torch
from diffusers import DiffusionPipeline

from . import config

_pipelines: dict[str, DiffusionPipeline] = {}


def load_pipeline(model_key: str) -> DiffusionPipeline:
    if model_key not in _pipelines:
        repo = config.MODELS[model_key]["repo"]
        pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=config.DTYPE)
        pipe.to(config.DEVICE)
        _pipelines[model_key] = pipe
    return _pipelines[model_key]


def generate_image(prompt: str, model_key: str, seed: int | None):
    if not prompt or not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if model_key not in config.MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Valid models: {', '.join(config.MODELS)}")

    pipe = load_pipeline(model_key)
    model_config = config.MODELS[model_key]

    generator = torch.Generator(device=config.DEVICE)
    if seed is not None and seed >= 0:
        resolved_seed = int(seed)
        generator = generator.manual_seed(resolved_seed)
    else:
        resolved_seed = generator.seed()

    start = time.time()
    result = pipe(
        prompt,
        num_inference_steps=model_config["steps"],
        guidance_scale=model_config["guidance_scale"],
        height=model_config["size"],
        width=model_config["size"],
        generator=generator,
    )
    elapsed = time.time() - start

    image = result.images[0]
    metadata = {
        "seed": resolved_seed,
        "width": model_config["size"],
        "height": model_config["size"],
        "elapsed_seconds": elapsed,
    }
    return image, metadata
