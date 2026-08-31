"""FastAPI app: route registration, static mounts, SPA fallback."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .pipelines import ensure_model_available
from .routes import generate, history, jobs, meta, models

logger = logging.getLogger(__name__)

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LiteDiffusion")


@app.on_event("startup")
def _prefetch_models() -> None:
    default_repo = config.MODELS[config.DEFAULT_MODEL]["repo"]
    logger.info("Prefetching default model: %s", default_repo)
    try:
        ensure_model_available(default_repo)
    except Exception as exc:
        logger.warning("Default model prefetch failed: %s", exc)


app.include_router(generate.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(meta.router, prefix="/api")
app.include_router(models.router, prefix="/api")

app.mount("/outputs", StaticFiles(directory=config.OUTPUTS_DIR), name="outputs")

if config.FRONTEND_DIST_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=config.FRONTEND_DIST_DIR / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        return FileResponse(config.FRONTEND_DIST_DIR / "index.html")
