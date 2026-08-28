"""Runtime device selection, model registry, and filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

MODELS = {
    "sd-turbo": {
        "label": "SD-Turbo — best quality (Recommended)",
        "repo": "stabilityai/sd-turbo",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
    },
    "sdxs-512": {
        "label": "SDXS-512 — fastest",
        "repo": "IDKiro/sdxs-512-0.9",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
    },
    "tiny-sd": {
        "label": "Tiny SD — smaller model, more steps",
        "repo": "segmind/tiny-sd",
        "steps": 20,
        "guidance_scale": 7.0,
        "size": 512,
    },
}
DEFAULT_MODEL = next(iter(MODELS))

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
INDEX_PATH = OUTPUTS_DIR / "index.json"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"
