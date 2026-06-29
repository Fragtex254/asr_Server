from __future__ import annotations

import pytest

from asr_server.adapters.qwen import QwenAsrAdapter
from asr_server.errors import AsrError


class OomModel:
    def transcribe(self, **kwargs: object) -> list[object]:
        del kwargs
        raise RuntimeError("CUDA out of memory")


class EmptyModel:
    def transcribe(self, **kwargs: object) -> list[object]:
        del kwargs
        return []


async def test_qwen_transcribe_maps_cuda_oom() -> None:
    adapter = QwenAsrAdapter("qwen3-asr-0.6b")
    adapter._model = OomModel()

    with pytest.raises(AsrError) as exc_info:
        await adapter.transcribe(
            b"audio",
            model_id="qwen3-asr-0.6b",
            backend="transformers",
            language="auto",
            context="",
            max_new_tokens=None,
        )

    assert exc_info.value.code == "gpu_unavailable"
    assert exc_info.value.details["phase"] == "inference"


async def test_qwen_transcribe_rejects_empty_results() -> None:
    adapter = QwenAsrAdapter("qwen3-asr-0.6b")
    adapter._model = EmptyModel()

    with pytest.raises(AsrError) as exc_info:
        await adapter.transcribe(
            b"audio",
            model_id="qwen3-asr-0.6b",
            backend="transformers",
            language="auto",
            context="",
            max_new_tokens=None,
        )

    assert exc_info.value.code == "inference_failed"
