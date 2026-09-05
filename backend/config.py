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
    # --- Local small/fast text-to-image (≤3GB each, CPU-friendly) ---
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
    "sdxl-turbo": {
        "label": "SDXL-Turbo — 1-step",
        "repo": "stabilityai/sdxl-turbo",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
        "kind": "image",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 2300,
        "pipeline": "StableDiffusionXLPipeline",
    },
    "dmd2": {
        "label": "DMD2 — 4-step SDXL",
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

    # --- Local text-to-video (prompt → video, no image required) ---
    "text-to-video-ms-1.7b": {
        "label": "AliV Lab Text-to-Video 1.7B — classic (Recommended)",
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

    # --- Local image-to-video (text + image → video; requires a reference image) ---
    "ltx-video-2b-i2v": {
        "label": "LTX-Video 2B (Recommended)",
        "repo": "Lightricks/LTX-Video",
        "steps": 30,
        "guidance_scale": 4.5,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 28000,
        "pipeline": "LTXImageToVideoPipeline",
        "frame_arg": "num_frames",
        "requires_prompt": True,
    },
    "i2vgen-xl": {
        "label": "I2VGen-XL — lightweight",
        "repo": "ali-vilab/i2vgen-xl",
        "steps": 50,
        "guidance_scale": 9.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 13000,
        "pipeline": "I2VGenXLPipeline",
        "frame_arg": "num_frames",
        "requires_prompt": True,
    },
    "sv3d": {
        "label": "SV3D — small, CPU-friendly",
        "repo": "stabilityai/sv3d",
        "steps": 50,
        "guidance_scale": 7.5,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 2500,
        "pipeline": "SV3DPipeline",
        "frame_arg": "num_frames",
        "requires_prompt": False,
    },
    "sv3d_humin3d": {
        "label": "SV3D-Human — optimized for people",
        "repo": "stabilityai/sv3d_humin3d",
        "steps": 50,
        "guidance_scale": 7.5,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 3200,
        "pipeline": "SV3DPipeline",
        "frame_arg": "num_frames",
        "requires_prompt": False,
    },

    # --- Local image-to-video: Stable Video Diffusion (small, CPU-friendly ~2GB) ---
    "svd": {
        "label": "SVD — Stable Video Diffusion (Recommended)",
        "repo": "stabilityai/svd",
        "steps": 25,
        "guidance_scale": 10.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 2200,
        "pipeline": "StableVideoDiffusionPipeline",
        "frame_arg": "num_frames",
        "requires_prompt": False,
    },
    "svd_xt": {
        "label": "SVD-XT — extended 14-frame, higher quality",
        "repo": "stabilityai/svd_xt",
        "steps": 25,
        "guidance_scale": 10.0,
        "size": 512,
        "kind": "video",
        "quantized": False,
        "remote": False,
        "provider": "local",
        "approx_size_mb": 2500,
        "pipeline": "StableVideoDiffusionPipeline",
        "frame_arg": "num_frames",
        "requires_prompt": False,
    },
}
DEFAULT_MODEL = "sdxs-512"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
INDEX_PATH = OUTPUTS_DIR / "index.json"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"

# Video settings: duration, fps, and default size for text-to-video generation.
# I2V-capable models accept an optional reference image through the /generate endpoint.
VIDEO = {
    "keyframes": 16,
    "fps": 12,
    "size": 512,
    "duration": 5,
}
