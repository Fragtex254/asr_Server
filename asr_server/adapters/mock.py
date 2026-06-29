from __future__ import annotations

import asyncio
import hashlib
from time import perf_counter

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings


class MockAsrAdapter:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.loaded_backend: str | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del device, dtype, max_new_tokens
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
        context: str,
        max_new_tokens: int | None,
    ) -> TranscriptionResult:
        inference_started = perf_counter()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        inference_ms = (perf_counter() - inference_started) * 1000
        return self._result(
            audio,
            model_id=model_id,
            backend=backend,
            language=language,
            context=context,
            max_new_tokens=max_new_tokens,
            inference_ms=inference_ms,
            batch=False,
            label=_audio_label(audio),
        )

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
        inference_started = perf_counter()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        inference_ms = (perf_counter() - inference_started) * 1000
        per_chunk_inference_ms = inference_ms / len(audio_chunks) if audio_chunks else 0.0
        return [
            self._result(
                audio,
                model_id=model_id,
                backend=backend,
                language=language,
                context=context,
                max_new_tokens=max_new_tokens,
                inference_ms=per_chunk_inference_ms,
                batch=True,
                label=_audio_label(audio),
            )
            for audio in audio_chunks
        ]

    def _result(
        self,
        audio: bytes,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
        inference_ms: float,
        batch: bool,
        label: str,
    ) -> TranscriptionResult:
        del model_id, backend
        text = f"{label}:{len(audio)}"
        warnings = ["mock_adapter"]
        if batch:
            warnings.append("mock_batch_adapter")
        if context:
            context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]
            warnings.append(f"context_received:{len(context)}:{context_hash}")
        if max_new_tokens is not None:
            warnings.append(f"max_new_tokens_received:{max_new_tokens}")
        return TranscriptionResult(
            text=text,
            duration=max(len(audio) / 16_000, 0.01),
            language="zh" if language == "auto" else language,
            warnings=warnings,
            timings=TranscriptionTimings(inference_ms=inference_ms),
        )


def _audio_label(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()[:12]
