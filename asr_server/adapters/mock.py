from __future__ import annotations

import asyncio

from asr_server.adapters.base import TranscriptionResult


class MockAsrAdapter:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.loaded_backend: str | None = None

    async def load(self, backend: str, device: str, dtype: str) -> None:
        del device, dtype
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        del cuda_empty_cache
        self.loaded_backend = None

    async def transcribe(
        self,
        audio: bytes,
        *,
        model_id: str,
        backend: str,
        language: str,
    ) -> TranscriptionResult:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        text = f"mock transcription for {model_id} using {backend}"
        if audio:
            text += f" ({len(audio)} bytes)"
        return TranscriptionResult(
            text=text,
            duration=max(len(audio) / 16_000, 0.01),
            language="zh" if language == "auto" else language,
            warnings=["mock_adapter"],
        )

