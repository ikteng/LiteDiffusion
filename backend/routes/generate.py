from __future__ import annotations

from flask import Blueprint, abort, jsonify, request
from pydantic import ValidationError

from .. import config, jobs
from ..models import GenerateRequest

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
