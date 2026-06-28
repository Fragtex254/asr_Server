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


def load_settings() -> Settings:
    adapter = os.getenv("ASR_ADAPTER", "mock")
    if adapter not in {"mock", "qwen"}:
        raise ValueError("ASR_ADAPTER must be 'mock' or 'qwen'")
    adapter_mode = cast(AdapterMode, adapter)
    return Settings(
        host=os.getenv("ASR_HOST", "0.0.0.0"),
        port=int(os.getenv("ASR_PORT", "18080")),
        public_base_url=os.getenv("ASR_PUBLIC_BASE_URL", "http://192.168.31.137:18080"),
        default_model=os.getenv("ASR_DEFAULT_MODEL", "qwen3-asr-1.7b"),
        adapter=adapter_mode,
    )
