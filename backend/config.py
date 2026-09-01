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
    # --- Local small/fast text-to-image (≤2GB each) ---
    "sdxs-512": {
        "label": "SDXS-512 — fastest (Recommended)",
        "repo": "IDKiro/sdxs-512-0.9",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 800,
    },
    "sdxs-256": {
        "label": "SDXS-256 — ultra fast",
        "repo": "IDKiro/sdxs-256-0.9",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 256,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 400,
    },
    "dmd2": {
        "label": "DMD2 4-step SDXL — painterly",
        "repo": "tianweiy/DMD2",
        "steps": 4,
        "guidance_scale": 0.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 1500,
    },
    "lcm-lora-sdxl": {
        "label": "LCM-LoRA SDXL — dreamy 1-2 step",
        "repo": "latent-consistency/lcm-lora-sdxl",
        "steps": 2,
        "guidance_scale": 1.5,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 1200,
        "pipeline": "LatentConsistencyModelPipeline",
    },
    "sana-0.6b": {
        "label": "Sana 0.6B — tiny + sharp",
        "repo": "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "steps": 8,
        "guidance_scale": 4.5,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 1200,
        "pipeline": "SanaPipeline",
    },
    "pixart-sigma-0.6b": {
        "label": "PixArt-Sigma 0.6B — small + detailed",
        "repo": "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        "steps": 6,
        "guidance_scale": 4.5,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 1300,
        "pipeline": "PixArtSigmaPipeline",
    },
    "tiny-sd": {
        "label": "Tiny SD — distilled Stable Diffusion",
        "repo": "segmind/tiny-sd",
        "steps": 15,
        "guidance_scale": 7.5,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 1060,
        "pipeline": "StableDiffusionPipeline",
    },
    "bk-sdm-tiny": {
        "label": "BK-SDM-Tiny — smallest SD architecture",
        "repo": "nota-ai/bk-sdm-tiny",
        "steps": 25,
        "guidance_scale": 7.5,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 950,
        "pipeline": "StableDiffusionPipeline",
    },

    # --- Local text-to-video (large downloads — T5-class text encoders dominate size) ---
    "ltx-video-2b": {
        "label": "LTX-Video 2B — fast",
        "repo": "Lightricks/LTX-Video",
        "steps": 30,
        "guidance_scale": 4.5,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 28000,
        "pipeline": "LTXPipeline",
        "frame_arg": "num_frames",
    },
    "ltx-video-distilled": {
        "label": "LTX-Video Distilled — 8-step fast",
        "repo": "Lightricks/LTX-Video-0.9.7-distilled",
        "steps": 8,
        "guidance_scale": 1.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 47600,
        "pipeline": "LTXPipeline",
        "frame_arg": "num_frames",
    },
    "text-to-video-ms-1.7b": {
        "label": "Text-to-Video 1.7B — classic (Recommended)",
        "repo": "ali-vilab/text-to-video-ms-1.7b",
        "steps": 50,
        "guidance_scale": 7.0,
        "size": 256,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 14700,
        "pipeline": "DiffusionPipeline",
        "frame_arg": "num_frames",
    },
    "kandinsky-5-lite-5s-distilled": {
        "label": "Kandinsky 5.0 Lite — distilled 16-step",
        "repo": "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers",
        "steps": 16,
        "guidance_scale": 5.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 23900,
        "pipeline": "Kandinsky5T2VPipeline",
        "frame_arg": "num_frames",
    },
    "kandinsky-5-lite-5s-nocfg": {
        "label": "Kandinsky 5.0 Lite — no-CFG",
        "repo": "kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers",
        "steps": 50,
        "guidance_scale": 5.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 23900,
        "pipeline": "Kandinsky5T2VPipeline",
        "frame_arg": "num_frames",
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
