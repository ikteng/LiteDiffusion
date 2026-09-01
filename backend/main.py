"""Flask app: route registration, static mounts, SPA fallback."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from . import config
from .routes.generate import bp as generate_bp
from .routes.history import bp as history_bp
from .routes.jobs import bp as jobs_bp
from .routes.meta import bp as meta_bp
from .routes.models import bp as models_bp

logger = logging.getLogger(__name__)

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)


# Startup prefetch is intentionally disabled — large models (e.g. sdxs-512 ~4GB,
# text-to-video ~7GB) were being re-downloaded on every dev-server restart.
# Trigger downloads explicitly via POST /api/models/{model_key}/download instead.


app.register_blueprint(generate_bp, url_prefix="/api")
app.register_blueprint(jobs_bp, url_prefix="/api")
app.register_blueprint(history_bp, url_prefix="/api")
app.register_blueprint(meta_bp, url_prefix="/api")
app.register_blueprint(models_bp, url_prefix="/api")


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    response = jsonify({"detail": exc.description})
    response.status_code = exc.code or 500
    return response


@app.get("/outputs/<path:filename>")
def serve_outputs(filename: str):
    return send_from_directory(config.OUTPUTS_DIR, filename)


if config.FRONTEND_DIST_DIR.exists():
    @app.get("/assets/<path:filename>")
    def serve_assets(filename: str):
        return send_from_directory(config.FRONTEND_DIST_DIR / "assets", filename)

    @app.get("/", defaults={"full_path": ""})
    @app.get("/<path:full_path>")
    def serve_spa(full_path: str):
        return send_from_directory(config.FRONTEND_DIST_DIR, "index.html")
