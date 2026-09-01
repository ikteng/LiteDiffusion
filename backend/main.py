"""FastAPI app: route registration, static mounts, SPA fallback."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .routes import generate, history, jobs, meta, models

logger = logging.getLogger(__name__)

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LiteDiffusion")


# Startup prefetch is intentionally disabled — large models (e.g. sdxs-512 ~4GB,
# text-to-video ~7GB) were being re-downloaded on every uvicorn --reload restart.
# Trigger downloads explicitly via POST /api/models/{model_key}/download instead.


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
