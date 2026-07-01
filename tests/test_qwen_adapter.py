from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from asr_server.adapters.qwen import QwenAsrAdapter, _QwenWorkerBackend
from asr_server.errors import AsrError


class OomModel:
    def transcribe(self, **kwargs: object) -> list[object]:
        del kwargs
        raise RuntimeError("CUDA out of memory")


class EmptyModel:
    def transcribe(self, **kwargs: object) -> list[object]:
        del kwargs
        return []


class TextResult:
    text = "ok"
    language = "en"


class RecordingModel:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def transcribe(self, **kwargs: object) -> list[TextResult]:
        del kwargs
        self.calls.append("transcribe")
        return [TextResult()]


class FakeWorker:
    def is_alive(self) -> bool:
        return True


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    def recv(self) -> dict[str, object]:
        request = self.sent[-1]
        return {
            "id": request["id"],
            "ok": True,
            "result": {
                "text": "worker text",
                "duration": 1.25,
                "language": "en",
                "warnings": ["from_worker"],
                "timings": {
                    "total_ms": 10.0,
                    "load_ms": 0.0,
                    "decode_ms": 1.0,
                    "inference_ms": 8.0,
                    "postprocess_ms": 1.0,
                },
            },
        }


class CleanupModel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ModelWithInnerModule(CleanupModel):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.model = SimpleNamespace(to=lambda device: calls.append(f"to:{device}"))


class FailingCleanupModel:
    def close(self) -> None:
        raise RuntimeError("close failed")


async def test_qwen_transcribe_maps_cuda_oom() -> None:
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
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
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
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


async def test_qwen_transcribe_uses_inference_mode_when_torch_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeInferenceMode:
        def __enter__(self) -> None:
            calls.append("enter_inference_mode")

        def __exit__(self, *args: object) -> None:
            calls.append("exit_inference_mode")

    fake_torch = SimpleNamespace(inference_mode=lambda: FakeInferenceMode())

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = RecordingModel(calls)

    result = await adapter.transcribe(
        b"audio",
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="",
        max_new_tokens=None,
    )

    assert result.text == "ok"
    assert calls == ["enter_inference_mode", "transcribe", "exit_inference_mode"]


async def test_qwen_transcribe_falls_back_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = RecordingModel(calls)

    result = await adapter.transcribe(
        b"audio",
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="",
        max_new_tokens=None,
    )

    assert result.text == "ok"
    assert calls == ["transcribe"]


async def test_qwen_adapter_transcribe_uses_worker_protocol() -> None:
    adapter = QwenAsrAdapter("qwen3-asr-0.6b")
    conn = FakeConnection()
    adapter._worker = FakeWorker()
    adapter._conn = conn  # type: ignore[assignment]

    result = await adapter.transcribe(
        b"audio",
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="ctx",
        max_new_tokens=128,
    )

    assert result.text == "worker text"
    assert result.timings.inference_ms == 8.0
    assert conn.sent == [
        {
            "id": 1,
            "op": "transcribe",
            "audio": b"audio",
            "language": "auto",
            "context": "ctx",
            "max_new_tokens": 128,
        }
    ]


async def test_qwen_unload_releases_model_and_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    model = CleanupModel()
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = model
    adapter.loaded_backend = "transformers"

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr("asr_server.adapters.qwen.gc.collect", lambda: calls.append("gc_collect"))
    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)

    await adapter.unload(cuda_empty_cache=True)

    assert model.closed is True
    assert getattr(adapter, "_model") is None
    assert getattr(adapter, "loaded_backend") is None
    assert calls == ["gc_collect", "synchronize", "empty_cache", "ipc_collect"]


async def test_qwen_unload_moves_inner_model_to_cpu_before_cuda_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = ModelWithInnerModule(calls)
    adapter.loaded_backend = "transformers"

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr("asr_server.adapters.qwen.gc.collect", lambda: calls.append("gc_collect"))
    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)

    await adapter.unload(cuda_empty_cache=True)

    assert calls == ["to:cpu", "gc_collect", "synchronize", "empty_cache", "ipc_collect"]


async def test_qwen_unload_continues_when_model_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = FailingCleanupModel()
    adapter.loaded_backend = "transformers"

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr("asr_server.adapters.qwen.gc.collect", lambda: calls.append("gc_collect"))
    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)

    await adapter.unload(cuda_empty_cache=True)

    assert getattr(adapter, "_model") is None
    assert getattr(adapter, "loaded_backend") is None
    assert calls == ["gc_collect", "synchronize", "empty_cache", "ipc_collect"]


async def test_qwen_unload_continues_when_cuda_synchronize_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = CleanupModel()
    adapter.loaded_backend = "transformers"

    def fail_synchronize() -> None:
        calls.append("synchronize")
        raise RuntimeError("sync failed")

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=fail_synchronize,
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr("asr_server.adapters.qwen.gc.collect", lambda: calls.append("gc_collect"))
    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)

    await adapter.unload(cuda_empty_cache=True)

    assert getattr(adapter, "_model") is None
    assert getattr(adapter, "loaded_backend") is None
    assert calls == ["gc_collect", "synchronize", "empty_cache", "ipc_collect"]
