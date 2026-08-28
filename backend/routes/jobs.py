from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import jobs as jobs_module
from ..models import JobResponse

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = jobs_module.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
