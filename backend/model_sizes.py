"""Resolves model download sizes: exact on-disk size once downloaded, otherwise
the manually-verified static `approx_size_mb` estimate from config.MODELS.

A live per-repo HF Hub API estimate was tried and dropped: many repos mix
duplicate .bin/.safetensors copies, fp16/fp32 variants, or (for non-diffusers
checkpoint repos like DMD2) dozens of unrelated training artifacts alongside
the single file actually used — this produced estimates off by 10-100x with no
generic way to filter them correctly. The static values are trustworthy because
each was checked by hand against the real files `from_pretrained` downloads.
"""

from __future__ import annotations

import logging

from huggingface_hub import scan_cache_dir

from . import config

logger = logging.getLogger(__name__)


def _on_disk_size_mb(repo: str) -> int | None:
    try:
        info = scan_cache_dir()
    except Exception:
        return None
    for cached_repo in info.repos:
        if cached_repo.repo_id == repo:
            return round(cached_repo.size_on_disk / 1e6)
    return None


def get_size_mb(model_key: str) -> int:
    model_config = config.MODELS[model_key]
    on_disk = _on_disk_size_mb(model_config["repo"])
    if on_disk is not None:
        return on_disk
    return model_config.get("approx_size_mb", 0)
