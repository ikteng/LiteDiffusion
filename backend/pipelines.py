"""Lazy-loaded diffusion pipelines and the core image/video generation calls."""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import time
from typing import Union

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import InferenceClient, snapshot_download
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

_pipelines: dict[str, DiffusionPipeline] = {}


def ensure_model_available(repo: str) -> None:
    try:
        snapshot_download(repo)
    except Exception as exc:
        logger.warning("Model prefetch failed for %s: %s", repo, exc)


def _get_remote_client() -> InferenceClient:
    api_key = config.REMOTE.get("api_key") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        logger.warning("No Hugging Face API key configured; remote inference may fail or be rate-limited.")
    return InferenceClient(api_key=api_key)


def _remote_generate_image(prompt: str, model_repo: str, size: int, seed: int | None) -> Image.Image:
    client = _get_remote_client()
    params = {
        "model": model_repo,
        "inputs": prompt,
        "parameters": {
            "width": size,
            "height": size,
            "seed": seed if seed is not None else 0,
        },
    }
    image_bytes = client.text_to_image(**params)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _remote_generate_video(prompt: str, model_repo: str, size: int, fps: int, frame_count: int) -> tuple[Path, int]:
    client = _get_remote_client()
    params = {
        "model": model_repo,
        "inputs": prompt,
        "parameters": {
            "width": size,
            "height": size,
            "num_frames": frame_count,
        },
    }
    video_bytes = client.text_to_video(**params)
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = config.VIDEOS_DIR / f"remote_{int(time.time() * 1000)}.mp4"
    with open(file_path, "wb") as f:
        f.write(video_bytes)
    reader = imageio.get_reader(str(file_path))
    frame_count = len(reader)
    reader.close()
    return file_path, frame_count


def load_pipeline(model_key: str) -> DiffusionPipeline:
    if model_key not in _pipelines:
        repo = config.MODELS[model_key]["repo"]
        model_config = config.MODELS[model_key]
        quantized = model_config.get("quantized", False)
        pipeline_name = model_config.get("pipeline", "DiffusionPipeline")
        logger.info("Loading pipeline for %s from %s (pipeline=%s, quantized=%s)", model_key, repo, pipeline_name, quantized)

        load_kwargs: dict = {"torch_dtype": config.DTYPE}

        if quantized and config.DEVICE == "cuda":
            try:
                import bitsandbytes  # noqa: F401 - optional CUDA quantization
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not installed; loading %s in full precision", model_key)

        if pipeline_name == "DiffusionPipeline":
            pipe = DiffusionPipeline.from_pretrained(repo, **load_kwargs)
        else:
            try:
                pipeline_cls = getattr(__import__("diffusers", fromlist=[pipeline_name]), pipeline_name)
                pipe = pipeline_cls.from_pretrained(repo, **load_kwargs)
            except (ImportError, AttributeError) as exc:
                logger.warning("Failed to load %s pipeline: %s. Falling back to DiffusionPipeline.", pipeline_name, exc)
                pipe = DiffusionPipeline.from_pretrained(repo, **load_kwargs)

        if quantized and config.DEVICE == "cuda" and "device_map" in load_kwargs:
            pipe.enable_model_cpu_offload()
        elif "device_map" not in load_kwargs:
            pipe.to(config.DEVICE)

        _pipelines[model_key] = pipe
    return _pipelines[model_key]


def generate_image(prompt: str, model_key: str, seed: int | None):
    if not prompt or not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if model_key not in config.MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Valid models: {', '.join(config.MODELS)}")

    model_config = config.MODELS[model_key]

    if model_config.get("remote") and config.REMOTE.get("enabled"):
        image = _remote_generate_image(prompt, model_config["repo"], model_config["size"], seed)
        metadata = {
            "seed": seed if seed is not None else 0,
            "width": model_config["size"],
            "height": model_config["size"],
            "elapsed_seconds": 0.0,
        }
        return image, metadata

    pipe = load_pipeline(model_key)

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


def _extract_frames(result) -> list[Image.Image]:
    if hasattr(result, "frames"):
        frames = result.frames
        if isinstance(frames, (list, tuple)) and frames:
            frames = frames[0]
    elif hasattr(result, "videos"):
        frames = result.videos
        if isinstance(frames, (list, tuple)) and frames:
            frames = frames[0]
    elif hasattr(result, "images"):
        frames = result.images
        if isinstance(frames, (list, tuple)) and frames:
            frames = frames[0]
    else:
        raise ValueError("Pipeline result has no frames, videos, or images")

    if hasattr(frames, "shape"):
        if len(frames.shape) == 4 and frames.shape[1] in (1, 3, 4):
            frames = [
                Image.fromarray(f.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                for f in frames
            ]
        elif len(frames.shape) == 4 and frames.shape[-1] in (1, 3, 4):
            frames = [
                Image.fromarray(f.cpu().numpy().astype(np.uint8))
                for f in frames
            ]
        else:
            raise ValueError(f"Unexpected frame tensor shape: {frames.shape}")
    else:
        frames = [
            f if hasattr(f, "save") else Image.fromarray(np.array(f))
            for f in frames
        ]

    return [f.convert("RGB") for f in frames]


def _save_frames_as_video(frames: list[Image.Image], file_path: Path, fps: int) -> int:
    with imageio.get_writer(
        str(file_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))
    return len(frames)


def generate_video(prompt: str, model_key: str, seed: int | None):
    if not prompt or not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if model_key not in config.MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Valid models: {', '.join(config.MODELS)}")

    model_config = config.MODELS[model_key]
    size = model_config.get("size", config.VIDEO["size"])
    fps = config.VIDEO["fps"]
    frame_count = config.VIDEO["keyframes"]

    if model_config.get("remote") and config.REMOTE.get("enabled"):
        file_path, frame_count = _remote_generate_video(prompt, model_config["repo"], size, fps, frame_count)
        elapsed = 0.0
        metadata = {
            "seed": seed if seed is not None else 0,
            "width": size,
            "height": size,
            "frames": frame_count,
            "fps": fps,
            "elapsed_seconds": elapsed,
        }
        return file_path, metadata

    pipe = load_pipeline(model_key)

    generator = torch.Generator(device=config.DEVICE)
    if seed is not None and seed >= 0:
        base_seed = int(seed)
    else:
        base_seed = generator.seed()

    start = time.time()

    if model_config.get("kind") == "video":
        frame_gen = torch.Generator(device=config.DEVICE).manual_seed(base_seed)
        result = pipe(
            prompt,
            num_inference_steps=model_config["steps"],
            guidance_scale=model_config["guidance_scale"],
            height=size,
            width=size,
            generator=frame_gen,
            num_frames=frame_count,
        )
        frames = _extract_frames(result)
        actual_fps = fps
        config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = config.VIDEOS_DIR / f"{abs(base_seed)}_{int(time.time() * 1000)}.mp4"
        frame_count = _save_frames_as_video(frames, file_path, actual_fps)
    else:
        keyframes: list[Image.Image] = []
        for i in range(frame_count):
            frame_seed = (base_seed + i) % (2**32)
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

            input_fps = frame_count
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(input_fps),
                "-i", os.path.join(tmpdir, "frame_%03d.png"),
                "-vf", f"minterpolate=fps={fps}:mi_mode=blend",
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
