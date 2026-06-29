from __future__ import annotations

import asyncio

import pytest

from asr_server.adapters.base import TranscriptionResult
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import default_models


class FailingLoadAdapter:
    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del backend, device, dtype, max_new_tokens
        raise RuntimeError("load failed")

    async def unload(self, cuda_empty_cache: bool) -> None:
        del cuda_empty_cache

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
        del audio, model_id, backend, language, context, max_new_tokens
        raise AssertionError("transcribe should not run when load fails")

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
        del audio_chunks, model_id, backend, language, context, max_new_tokens
        raise AssertionError("transcribe_batch should not run when load fails")


class RecordingAdapter:
    def __init__(self) -> None:
        self.loads: list[int | None] = []
        self.active_transcribes = 0
        self.max_parallel_transcribes = 0
        self.unload_while_transcribing = False

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del backend, device, dtype
        self.loads.append(max_new_tokens)

    async def unload(self, cuda_empty_cache: bool) -> None:
        del cuda_empty_cache
        if self.active_transcribes > 0:
            self.unload_while_transcribing = True

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
        del audio, model_id, backend, context, max_new_tokens
        self.active_transcribes += 1
        self.max_parallel_transcribes = max(self.max_parallel_transcribes, self.active_transcribes)
        await asyncio.sleep(0.03)
        self.active_transcribes -= 1
        return TranscriptionResult(text="ok", duration=1.0, language="zh" if language == "auto" else language, warnings=[])

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
        del audio_chunks, model_id, backend, language, context, max_new_tokens
        raise AssertionError("batch path is not used in this test")


async def test_load_failure_sets_model_error_state() -> None:
    manager = ModelLifecycleManager(default_models(), lambda _model_id: FailingLoadAdapter())

    with pytest.raises(RuntimeError, match="load failed"):
        await manager.load_model("qwen3-asr-1.7b", backend="transformers")

    assert manager.runtime_for("qwen3-asr-1.7b").status == "error"


async def test_same_model_transcriptions_are_serialized_across_generation_reload() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter)

    first = asyncio.create_task(
        manager.transcribe_chunks(
            [b"first"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=512,
            batch_size=1,
        )
    )
    await asyncio.sleep(0.01)
    second = asyncio.create_task(
        manager.transcribe_chunks(
            [b"second"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=256,
            batch_size=1,
        )
    )

    await asyncio.gather(first, second)

    assert adapter.loads == [512, 256]
    assert adapter.max_parallel_transcribes == 1
    assert adapter.unload_while_transcribing is False
