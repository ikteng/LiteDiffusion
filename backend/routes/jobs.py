from __future__ import annotations

from flask import Blueprint, abort, jsonify

from .. import jobs as jobs_module

bp = Blueprint("jobs", __name__)


@bp.get("/jobs/<job_id>")
def get_job(job_id: str):
    job = jobs_module.get_job(job_id)
    if job is None:
        abort(404, description="Job not found")
    return jsonify(job.model_dump(mode="json"))


@bp.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    jobs_module.cancel_job(job_id)
    return jsonify({"status": "cancelled"})
