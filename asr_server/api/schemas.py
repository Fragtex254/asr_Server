from __future__ import annotations

from pydantic import BaseModel

from asr_server.registry import Backend


class LoadRequest(BaseModel):
    backend: Backend = "auto"
    device: str = "cuda"
    dtype: str = "auto"


class UnloadRequest(BaseModel):
    mode: str = "after_current_requests"
    reject_new_requests: bool = True
    cuda_empty_cache: bool = True
