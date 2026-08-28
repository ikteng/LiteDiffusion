"""FastAPI app: route registration, static mounts, SPA fallback."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .routes import generate, history, jobs, meta

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LiteDiffusion")

app.include_router(generate.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(meta.router, prefix="/api")

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
