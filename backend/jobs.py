"""In-memory job registry and single-worker background execution."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import config, pipelines, store
from .models import GenerateRequest, HistoryItem, JobResponse, JobResult, JobStatus, MediaType

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, JobResponse] = {}


def get_job(job_id: str) -> JobResponse | None:
    return _jobs.get(job_id)


def submit_job(request: GenerateRequest) -> JobResponse:
    job_id = uuid.uuid4().hex
    job = JobResponse(
        id=job_id,
        status=JobStatus.QUEUED,
        media_type=request.media_type,
        prompt=request.prompt,
        model=request.model,
        created_at=time.time(),
    )
    _jobs[job_id] = job
    _executor.submit(_run_job, job_id, request)
    return job


def _run_job(job_id: str, request: GenerateRequest) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING

    try:
        image, metadata = pipelines.generate_image(request.prompt, request.model, request.seed)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as job.error
        job.status = JobStatus.FAILED
        job.error = str(exc)
        return

    filename = f"{job_id}.png"
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image.save(config.IMAGES_DIR / filename)
    relative_file = f"images/{filename}"

    result = JobResult(
        media_type=MediaType.IMAGE,
        file_url=f"/outputs/{relative_file}",
        width=metadata["width"],
        height=metadata["height"],
        seed=metadata["seed"],
        elapsed_seconds=metadata["elapsed_seconds"],
    )

    history_item = HistoryItem(
        id=job_id,
        media_type=MediaType.IMAGE,
        prompt=request.prompt,
        model=request.model,
        seed=metadata["seed"],
        width=metadata["width"],
        height=metadata["height"],
        elapsed_seconds=metadata["elapsed_seconds"],
        created_at=job.created_at,
        file=relative_file,
        file_url=result.file_url,
    )
    store.append_history(history_item)

    job.result = result
    job.status = JobStatus.SUCCEEDED
