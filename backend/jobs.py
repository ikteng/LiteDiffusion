"""In-memory job registry and single-worker background execution."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import abort

from . import config, pipelines, store
from .models import GenerateRequest, HistoryItem, JobResponse, JobResult, JobStatus, MediaType

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, JobResponse] = {}
_cancel_events: dict[str, threading.Event] = {}


def get_job(job_id: str) -> JobResponse | None:
    return _jobs.get(job_id)


def cancel_job(job_id: str) -> None:
    if job_id not in _jobs:
        abort(404, description="Job not found")
    job = _jobs[job_id]
    if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
        abort(400, description="Job cannot be cancelled")
    event = _cancel_events.get(job_id)
    if event is None:
        event = threading.Event()
        _cancel_events[job_id] = event
    event.set()
    job.status = JobStatus.FAILED
    job.error = "Cancelled by user"


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
    _cancel_events[job_id] = threading.Event()
    _executor.submit(_run_job, job_id, request)
    return job


def _run_job(job_id: str, request: GenerateRequest) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING

    try:
        if request.media_type == MediaType.VIDEO:
            file_path, metadata = pipelines.generate_video(
                request.prompt,
                request.model,
                request.seed,
                request.duration,
                _cancel_events[job_id],
                reference_image=request.reference_image,
                end_image=request.end_image,
            )
            filename = file_path.name
            relative_file = f"videos/{filename}"
            result = JobResult(
                media_type=MediaType.VIDEO,
                file_url=f"/outputs/{relative_file}",
                width=metadata["width"],
                height=metadata["height"],
                seed=metadata["seed"],
                elapsed_seconds=metadata["elapsed_seconds"],
                frames=metadata["frames"],
                fps=metadata["fps"],
            )
        else:
            image, metadata = pipelines.generate_image(
                request.prompt,
                request.model,
                request.seed,
                _cancel_events[job_id],
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
            )
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
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as job.error
        job.status = JobStatus.FAILED
        job.error = str(exc)
        return

    history_item = HistoryItem(
        id=job_id,
        media_type=result.media_type,
        prompt=request.prompt,
        model=request.model,
        seed=result.seed,
        width=result.width,
        height=result.height,
        elapsed_seconds=result.elapsed_seconds,
        created_at=job.created_at,
        file=relative_file,
        file_url=result.file_url,
        frames=result.frames,
        fps=result.fps,
    )
    store.append_history(history_item)

    job.result = result
    job.status = JobStatus.SUCCEEDED
