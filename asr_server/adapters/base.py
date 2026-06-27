from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration: float
    language: str
    warnings: list[str]


class AsrAdapter(Protocol):
    async def load(self, backend: str, device: str, dtype: str) -> None:
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
    ) -> TranscriptionResult:
        ...

