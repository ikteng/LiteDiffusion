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
    if request.is_json:
        payload = GenerateRequest.model_validate(request.get_json(force=True))
    else:
        form = request.form.to_dict()
        payload = GenerateRequest.model_validate(
            {
                "prompt": form.get("prompt", ""),
                "model": form.get("model", ""),
                "seed": int(form["seed"]) if form.get("seed") not in (None, "") else None,
                "media_type": form.get("media_type") or "image",
                "duration": int(form["duration"]) if form.get("duration") not in (None, "") else None,
                "negative_prompt": form.get("negative_prompt") or None,
                "steps": int(form["steps"]) if form.get("steps") not in (None, "") else None,
                "guidance_scale": float(form["guidance_scale"]) if form.get("guidance_scale") not in (None, "") else None,
            }
        )

        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        model_key = payload.model

        upload = request.files.get("image")
        if upload and upload.filename:
            if not (upload.mimetype or "").startswith("image/"):
                abort(422, description="Uploaded file must be an image.")
            ext = Path(upload.filename).suffix or ".png"
            saved_path = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
            upload.save(saved_path)
            payload.reference_image = str(saved_path)

        end_upload = request.files.get("end_image")
        if end_upload and end_upload.filename:
            if not (end_upload.mimetype or "").startswith("image/"):
                abort(422, description="Uploaded end frame must be an image.")
            if config.MODELS[model_key].get("pipeline") != "LTXImageToVideoPipeline":
                abort(400, description="First/last-frame mode is only supported by LTX-Video.")
            ext = Path(end_upload.filename).suffix or ".png"
            end_saved_path = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
            end_upload.save(end_saved_path)
            payload.end_image = str(end_saved_path)

    if payload.model not in config.MODELS:
        valid = ", ".join(config.MODELS)
        abort(400, description=f"Unknown model '{payload.model}'. Valid models: {valid}")

    job = jobs.submit_job(payload)
    return jsonify(job.model_dump(mode="json")), 202
