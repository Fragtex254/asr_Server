from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

AdapterMode = Literal["mock", "qwen"]


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 18080
    public_base_url: str = "http://192.168.31.137:18080"
    default_model: str = "qwen3-asr-1.7b"
    adapter: AdapterMode = "mock"
    qwen_batch_size: int = 1
    job_result_ttl_seconds: int = 3600
    idle_unload_seconds: float = 180.0
    max_queued_jobs: int = 20
    max_upload_mb: int = 512


def load_settings() -> Settings:
    adapter = os.getenv("ASR_ADAPTER", "mock")
    if adapter not in {"mock", "qwen"}:
        raise ValueError("ASR_ADAPTER must be 'mock' or 'qwen'")
    adapter_mode = cast(AdapterMode, adapter)
    qwen_batch_size = int(os.getenv("ASR_QWEN_BATCH_SIZE", "1"))
    if qwen_batch_size < 1:
        raise ValueError("ASR_QWEN_BATCH_SIZE must be greater than or equal to 1")
    job_result_ttl_seconds = int(os.getenv("ASR_JOB_RESULT_TTL_SECONDS", "3600"))
    if job_result_ttl_seconds < 1:
        raise ValueError("ASR_JOB_RESULT_TTL_SECONDS must be greater than or equal to 1")
    idle_unload_seconds = float(os.getenv("ASR_IDLE_UNLOAD_SECONDS", "180"))
    if idle_unload_seconds < 0:
        raise ValueError("ASR_IDLE_UNLOAD_SECONDS must be greater than or equal to 0")
    max_queued_jobs = int(os.getenv("ASR_MAX_QUEUED_JOBS", "20"))
    if max_queued_jobs < 1:
        raise ValueError("ASR_MAX_QUEUED_JOBS must be greater than or equal to 1")
    max_upload_mb = int(os.getenv("ASR_MAX_UPLOAD_MB", "512"))
    if max_upload_mb < 1:
        raise ValueError("ASR_MAX_UPLOAD_MB must be greater than or equal to 1")
    return Settings(
        host=os.getenv("ASR_HOST", "0.0.0.0"),
        port=int(os.getenv("ASR_PORT", "18080")),
        public_base_url=os.getenv("ASR_PUBLIC_BASE_URL", "http://192.168.31.137:18080"),
        default_model=os.getenv("ASR_DEFAULT_MODEL", "qwen3-asr-1.7b"),
        adapter=adapter_mode,
        qwen_batch_size=qwen_batch_size,
        job_result_ttl_seconds=job_result_ttl_seconds,
        idle_unload_seconds=idle_unload_seconds,
        max_queued_jobs=max_queued_jobs,
        max_upload_mb=max_upload_mb,
    )
