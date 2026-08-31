"""Runtime device selection, model registry, and filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

REMOTE = {
    "enabled": os.getenv("LITEDIFFUSION_REMOTE", "false").lower() == "true",
    "api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
    "text_to_image": os.getenv("LITEDIFFUSION_REMOTE_T2I", "stabilityai/stable-diffusion-xl-base-1.0"),
    "text_to_video": os.getenv("LITEDIFFUSION_REMOTE_T2V", "ali-vilab/text-to-video-ms-1.7b"),
}

MODELS = {
    "sdxs-512": {
        "label": "SDXS-512 — fastest",
        "repo": "IDKiro/sdxs-512-0.9",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
    },
    "sd-turbo": {
        "label": "SD-Turbo — best quality (Recommended)",
        "repo": "stabilityai/sd-turbo",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
    },
    "text-to-video-ms-1.7b": {
        "label": "Text-to-Video 1.7B — local T2V",
        "repo": "ali-vilab/text-to-video-ms-1.7b",
        "steps": 50,
        "guidance_scale": 7.0,
        "size": 256,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "pipeline": "DiffusionPipeline",
        "frame_arg": "num_frames",
    },
    "kandinsky-5-lite-5s-distilled": {
        "label": "Kandinsky 5.0 Lite 5s Distilled — local T2V (2B)",
        "repo": "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers",
        "steps": 16,
        "guidance_scale": 5.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "pipeline": "Kandinsky5T2VPipeline",
        "frame_arg": "num_frames",
    },
    "kandinsky-5-lite-5s-nocfg": {
        "label": "Kandinsky 5.0 Lite 5s No-CFG — local T2V (2B)",
        "repo": "kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers",
        "steps": 50,
        "guidance_scale": 5.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "pipeline": "Kandinsky5T2VPipeline",
        "frame_arg": "num_frames",
    },
    "remote-t2i-sdxl": {
        "label": "SDXL Base 1.0 — remote T2I",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": 1,
        "guidance_scale": 7.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": True,
    },
    "remote-t2i-sd15": {
        "label": "SD 1.5 — remote T2I",
        "repo": "runwayml/stable-diffusion-v1-5",
        "steps": 1,
        "guidance_scale": 7.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": True,
    },
    "remote-t2v-1.7b": {
        "label": "Text-to-Video 1.7B — remote T2V",
        "repo": "ali-vilab/text-to-video-ms-1.7b",
        "steps": 1,
        "guidance_scale": 7.0,
        "size": 256,
        "kind": "video",
        "quantized": False,
        "remote": True,
    },
    "remote-t2v-animatediff": {
        "label": "AnimateDiff — remote T2V",
        "repo": "guoyww/animatediff",
        "steps": 1,
        "guidance_scale": 7.0,
        "size": 256,
        "kind": "video",
        "quantized": False,
        "remote": True,
    },
}
DEFAULT_MODEL = "sdxs-512"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
INDEX_PATH = OUTPUTS_DIR / "index.json"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"

# Text-to-video via keyframe generation + frame interpolation. Reuses the image
# models (already loaded) so no extra weights are needed; motion is a smooth
# morph between a few generated keyframes. Real T2V models (kind=video) bypass
# keyframe mode and generate frames directly. Remote models use the Hugging Face
# Inference API and do not download weights.
VIDEO = {
    "keyframes": 16,
    "fps": 12,
    "size": 512,
    "duration": 5,
}
