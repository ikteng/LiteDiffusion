from __future__ import annotations

import logging
import threading
from typing import Dict

from flask import Blueprint, abort, jsonify

from .. import config
from ..pipelines import ensure_model_available

logger = logging.getLogger(__name__)

bp = Blueprint("models", __name__)

_download_status: Dict[str, str] = {}
_download_lock = threading.Lock()


@bp.post("/models/<model_key>/download")
def download_model(model_key: str):
    if model_key not in config.MODELS:
        abort(404, description=f"Unknown model '{model_key}'")

    model_config = config.MODELS[model_key]
    if model_config.get("provider", "local") != "local":
        abort(400, description="Remote / non-local models do not require download")

    with _download_lock:
        status = _download_status.get(model_key)
        if status == "downloading":
            return jsonify({"status": "downloading"})
        if status == "ready":
            return jsonify({"status": "ready"})

        _download_status[model_key] = "downloading"

    def _do_download() -> None:
        try:
            ensure_model_available(model_config["repo"])
            with _download_lock:
                _download_status[model_key] = "ready"
        except Exception as exc:
            logger.error("Model download failed for %s: %s", model_key, exc)
            with _download_lock:
                _download_status[model_key] = f"error: {exc}"

    threading.Thread(target=_do_download, daemon=True).start()
    return jsonify({"status": "downloading"})


@bp.get("/models/status")
def get_model_status():
    with _download_lock:
        return jsonify(dict(_download_status))
