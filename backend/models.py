"""Pydantic request/response schemas shared across routes."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    prompt: str = ""
    model: str
    seed: int | None = None
    media_type: MediaType = MediaType.IMAGE
    duration: int | None = None
    negative_prompt: str | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    reference_image: str | None = None
    end_image: str | None = None


class JobResult(BaseModel):
    media_type: MediaType
    file_url: str
    width: int
    height: int
    seed: int
    elapsed_seconds: float
    frames: int | None = None
    fps: float | None = None


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    media_type: MediaType
    prompt: str
    model: str
    created_at: float
    error: str | None = None
    result: JobResult | None = None


class HistoryItem(BaseModel):
    id: str
    media_type: MediaType
    prompt: str
    model: str
    seed: int
    width: int
    height: int
    elapsed_seconds: float
    created_at: float
    file: str
    file_url: str
    frames: int | None = None
    fps: float | None = None


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class ModelInfo(BaseModel):
    key: str
    label: str
    repo: str
    steps: int
    guidance_scale: float
    size: int
    kind: str = "image"
    quantized: bool = False
    pipeline: str = "DiffusionPipeline"
    frame_arg: str = "num_frames"
    remote: bool = False
    provider: str = "local"
    approx_size_mb: int = 0
    requires_prompt: bool = True


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    default: str


class SettingsResponse(BaseModel):
    device: str
    dtype: str
