from __future__ import annotations

from flask import Blueprint, jsonify

from .. import config, model_sizes

bp = Blueprint("meta", __name__)


@bp.get("/models")
def list_models():
    models = [
        {
            "key": key,
            "label": value["label"],
            "repo": value["repo"],
            "steps": value["steps"],
            "guidance_scale": value["guidance_scale"],
            "size": value["size"],
            "kind": value.get("kind", "image"),
            "quantized": value.get("quantized", False),
            "pipeline": value.get("pipeline", "DiffusionPipeline"),
            "frame_arg": value.get("frame_arg", "num_frames"),
            "remote": value.get("remote", False),
            "provider": value.get("provider", "local"),
            "approx_size_mb": model_sizes.get_size_mb(key) if not value.get("remote") else 0,
        }
        for key, value in config.MODELS.items()
    ]
    return jsonify({"models": models, "default": config.DEFAULT_MODEL})


@bp.get("/settings")
def get_settings():
    dtype = "float16" if config.DEVICE == "cuda" else "float32"
    return jsonify({"device": config.DEVICE, "dtype": dtype})
