from __future__ import annotations

import asyncio
from time import perf_counter

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings


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
        inference_started = perf_counter()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        inference_ms = (perf_counter() - inference_started) * 1000
        text = f"mock transcription for {model_id} using {backend}"
        if audio:
            text += f" ({len(audio)} bytes)"
        return TranscriptionResult(
            text=text,
            duration=max(len(audio) / 16_000, 0.01),
            language="zh" if language == "auto" else language,
            warnings=["mock_adapter"],
            timings=TranscriptionTimings(inference_ms=inference_ms),
        )
