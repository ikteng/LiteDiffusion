from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .. import config, jobs
from ..models import GenerateRequest, JobResponse

router = APIRouter()


@router.post("/generate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def generate(request: GenerateRequest) -> JobResponse:
    if request.model not in config.MODELS:
        valid = ", ".join(config.MODELS)
        raise HTTPException(status_code=400, detail=f"Unknown model '{request.model}'. Valid models: {valid}")

    return jobs.submit_job(request)
