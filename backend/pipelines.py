"""Lazy-loaded diffusion pipelines and the core image/video generation calls."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

_pipelines: dict[str, DiffusionPipeline] = {}


def ensure_model_available(repo: str) -> None:
    try:
        snapshot_download(repo)
    except Exception as exc:
        logger.warning("Model prefetch failed for %s: %s", repo, exc)


def load_pipeline(model_key: str) -> DiffusionPipeline:
    if model_key not in _pipelines:
        repo = config.MODELS[model_key]["repo"]
        logger.info("Loading pipeline for %s from %s", model_key, repo)
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


def generate_video(prompt: str, model_key: str, seed: int | None):
    if not prompt or not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if model_key not in config.MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Valid models: {', '.join(config.MODELS)}")

    pipe = load_pipeline(model_key)
    model_config = config.MODELS[model_key]
    size = config.VIDEO["size"]
    keyframes_count = config.VIDEO["keyframes"]
    fps = config.VIDEO["fps"]

    generator = torch.Generator(device=config.DEVICE)
    if seed is not None and seed >= 0:
        base_seed = int(seed)
    else:
        base_seed = generator.seed()

    start = time.time()
    keyframes: list[Image.Image] = []
    for i in range(keyframes_count):
        frame_seed = (base_seed + i * 1_000_003) % (2**32)
        frame_gen = torch.Generator(device=config.DEVICE).manual_seed(frame_seed)
        out = pipe(
            prompt,
            num_inference_steps=model_config["steps"],
            guidance_scale=model_config["guidance_scale"],
            height=size,
            width=size,
            generator=frame_gen,
        )
        keyframes.append(out.images[0].convert("RGB"))

    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = config.VIDEOS_DIR / f"{abs(base_seed)}_{int(time.time() * 1000)}.mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, kf in enumerate(keyframes):
            kf.save(os.path.join(tmpdir, f"frame_{i:03d}.png"))

        input_fps = max(keyframes_count // 2, 1)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_fps),
            "-i", os.path.join(tmpdir, "frame_%03d.png"),
            "-vf", f"minterpolate=fps={fps}:mi_mode=mc:mc_mode=aobmc",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(file_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    reader = imageio.get_reader(str(file_path))
    frame_count = len(reader)
    reader.close()

    elapsed = time.time() - start

    metadata = {
        "seed": base_seed,
        "width": size,
        "height": size,
        "frames": frame_count,
        "fps": fps,
        "elapsed_seconds": elapsed,
    }
    return file_path, metadata
