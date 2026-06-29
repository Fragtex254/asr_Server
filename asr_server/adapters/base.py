from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionTimings:
    total_ms: float = 0.0
    load_ms: float = 0.0
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def to_api(self) -> dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "load_ms": self.load_ms,
            "decode_ms": self.decode_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration: float
    language: str
    warnings: list[str]
    timings: TranscriptionTimings = field(default_factory=TranscriptionTimings)


class AsrAdapter(Protocol):
    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        ...

    async def unload(self, cuda_empty_cache: bool) -> None:
        ...

    async def transcribe(
        self,
        audio: bytes,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> TranscriptionResult:
        ...

    async def transcribe_batch(
        self,
        audio_chunks: list[bytes],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> list[TranscriptionResult]:
        ...
