from __future__ import annotations

from fastapi import APIRouter

from .. import config
from ..models import ModelInfo, ModelListResponse, SettingsResponse

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    models = [
        ModelInfo(
            key=key,
            label=value["label"],
            repo=value["repo"],
            steps=value["steps"],
            guidance_scale=value["guidance_scale"],
            size=value["size"],
        )
        for key, value in config.MODELS.items()
    ]
    return ModelListResponse(models=models, default=config.DEFAULT_MODEL)


@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    dtype = "float16" if config.DEVICE == "cuda" else "float32"
    return SettingsResponse(device=config.DEVICE, dtype=dtype)
