from __future__ import annotations

import logging
import threading
from typing import Dict

from fastapi import APIRouter, HTTPException

from .. import config
from ..pipelines import ensure_model_available

logger = logging.getLogger(__name__)

router = APIRouter()

_download_status: Dict[str, str] = {}
_download_lock = threading.Lock()


@router.post("/models/{model_key}/download")
def download_model(model_key: str) -> Dict[str, str]:
    if model_key not in config.MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'")

    model_config = config.MODELS[model_key]
    if model_config.get("remote", False):
        raise HTTPException(status_code=400, detail="Remote models do not require download")

    with _download_lock:
        status = _download_status.get(model_key)
        if status == "downloading":
            return {"status": "downloading"}
        if status == "ready":
            return {"status": "ready"}

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
    return {"status": "downloading"}


@router.get("/models/status")
def get_model_status() -> Dict[str, str]:
    with _download_lock:
        return dict(_download_status)
