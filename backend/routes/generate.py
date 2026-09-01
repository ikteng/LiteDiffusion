from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, abort, jsonify, request
from pydantic import ValidationError

from .. import config, jobs
from ..models import GenerateRequest, MediaType

bp = Blueprint("generate", __name__)


@bp.post("/generate")
def generate():
    try:
        payload = GenerateRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        abort(422, description=exc.errors())

    if payload.model not in config.MODELS:
        valid = ", ".join(config.MODELS)
        abort(400, description=f"Unknown model '{payload.model}'. Valid models: {valid}")

    job = jobs.submit_job(payload)
    return jsonify(job.model_dump(mode="json")), 202


@bp.post("/generate/image-to-video")
def generate_image_to_video():
    model_key = request.form.get("model", "")
    if model_key not in config.MODELS or config.MODELS[model_key].get("kind") != "image_to_video":
        valid = ", ".join(k for k, m in config.MODELS.items() if m.get("kind") == "image_to_video")
        abort(400, description=f"Unknown image-to-video model '{model_key}'. Valid models: {valid}")

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        abort(422, description="An image file is required.")
    if not (upload.mimetype or "").startswith("image/"):
        abort(422, description="Uploaded file must be an image.")

    end_upload = request.files.get("end_image")
    if end_upload is not None and end_upload.filename:
        if not (end_upload.mimetype or "").startswith("image/"):
            abort(422, description="Uploaded end frame must be an image.")
        if config.MODELS[model_key].get("pipeline") != "LTXImageToVideoPipeline":
            abort(400, description="First/last-frame mode is only supported by LTX-Video.")
    else:
        end_upload = None

    seed_raw = request.form.get("seed")
    duration_raw = request.form.get("duration")

    try:
        payload = GenerateRequest.model_validate(
            {
                "prompt": request.form.get("prompt", ""),
                "model": model_key,
                "seed": int(seed_raw) if seed_raw not in (None, "") else None,
                "media_type": MediaType.VIDEO,
                "duration": int(duration_raw) if duration_raw not in (None, "") else None,
            }
        )
    except ValidationError as exc:
        abort(422, description=exc.errors())

    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename).suffix or ".png"
    saved_path = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    upload.save(saved_path)
    payload.reference_image = str(saved_path)

    if end_upload is not None:
        end_ext = Path(end_upload.filename).suffix or ".png"
        end_saved_path = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{end_ext}"
        end_upload.save(end_saved_path)
        payload.end_image = str(end_saved_path)

    job = jobs.submit_job(payload)
    return jsonify(job.model_dump(mode="json")), 202
