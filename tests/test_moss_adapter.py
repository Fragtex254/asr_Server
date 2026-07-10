from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from asr_server.adapters.base import TranscriptionSegment
from asr_server.adapters.moss import MODEL_REPOS, MossTranscribeDiarizeAdapter, _MossHelpers, _MossWorkerBackend
from asr_server.errors import AsrError


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
                "language": "auto",
                "warnings": [],
                "segments": [
                    {
                        "start": 0.1,
                        "end": 0.6,
                        "speaker": "S01",
                        "text": "worker text",
                    }
                ],
                "timings": {
                    "total_ms": 10.0,
                    "load_ms": 0.0,
                    "decode_ms": 1.0,
                    "inference_ms": 8.0,
                    "postprocess_ms": 1.0,
                },
            },
        }


class FakeTransport:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def request(self, op: str, *, timeout_seconds: float, **payload: object) -> object:
        del timeout_seconds
        self.conn.send({"id": len(self.conn.sent) + 1, "op": op, **payload})
        return self.conn.recv()["result"]


def test_moss_model_repo_uses_fixed_hf_model_id() -> None:
    assert MODEL_REPOS == {
        "moss-transcribe-diarize-0.9b": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    }


async def test_moss_load_and_transcribe_use_official_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str, trust_remote_code: bool) -> "FakeProcessor":
            calls.append(("processor_repo", repo_id))
            calls.append(("processor_revision", revision))
            calls.append(("processor_trust_remote_code", trust_remote_code))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str, trust_remote_code: bool, dtype: object) -> "FakeModel":
            calls.append(("model_repo", repo_id))
            calls.append(("model_revision", revision))
            calls.append(("model_trust_remote_code", trust_remote_code))
            calls.append(("model_dtype", dtype))
            return cls()

        def to(self, *args: object, **kwargs: object) -> "FakeModel":
            if "dtype" in kwargs:
                calls.append(("to_dtype", kwargs["dtype"]))
            elif args:
                calls.append(("to_device", args[0]))
            return self

        def eval(self) -> "FakeModel":
            calls.append(("eval", True))
            return self

    fake_device = SimpleNamespace(type="cuda")

    def fake_torch_device(name: str) -> object:
        calls.append(("torch_device", name))
        return fake_device

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True),
        device=fake_torch_device,
    )
    fake_transformers = SimpleNamespace(AutoProcessor=FakeProcessor, AutoModelForCausalLM=FakeModel)

    def build_transcription_messages(audio_path: str, *, prompt: str) -> list[dict[str, object]]:
        calls.append(("audio_path_suffix", audio_path.endswith(".wav")))
        calls.append(("prompt", prompt))
        return [{"role": "user", "content": prompt}]

    def generate_transcription(
        model: object,
        processor: object,
        messages: object,
        *,
        max_new_tokens: int,
        do_sample: bool,
        device: object,
        dtype: object,
    ) -> dict[str, str]:
        del model, processor, messages
        calls.append(("generate_max_new_tokens", max_new_tokens))
        calls.append(("generate_do_sample", do_sample))
        calls.append(("generate_device", device))
        calls.append(("generate_dtype", dtype))
        return {"text": "[0.10][S01]hello[0.60]"}

    def parse_transcript(text: str) -> list[SimpleNamespace]:
        calls.append(("parse_text", text))
        return [SimpleNamespace(start=0.0, end=0.001, speaker="S01", text="hello")]

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        if name == "moss_transcribe_diarize":
            return SimpleNamespace(parse_transcript=parse_transcript)
        if name == "moss_transcribe_diarize.inference_utils":
            return SimpleNamespace(
                DEFAULT_PROMPT="default prompt",
                build_transcription_messages=build_transcription_messages,
                generate_transcription=generate_transcription,
            )
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("asr_server.adapters.moss.importlib.import_module", fake_import_module)
    adapter = _MossWorkerBackend("moss-transcribe-diarize-0.9b")

    await adapter.load("transformers", "cuda", "auto", max_new_tokens=2048)
    result = await adapter.transcribe(
        b"audio",
        model_id="moss-transcribe-diarize-0.9b",
        backend="transformers",
        language="auto",
        context="术语：MOSS\n热词提示：RTX 5070 Ti",
        max_new_tokens=1024,
    )

    assert result.text == "hello"
    assert result.language == "auto"
    assert result.segments[0].speaker == "S01"
    assert result.segments[0].start == 0.0
    assert ("model_repo", "OpenMOSS-Team/MOSS-Transcribe-Diarize") in calls
    assert ("model_trust_remote_code", True) in calls
    assert ("model_dtype", "auto") in calls
    assert ("processor_trust_remote_code", True) in calls
    assert ("to_dtype", "bf16") in calls
    assert ("to_device", fake_device) in calls
    assert ("generate_max_new_tokens", 1024) in calls
    assert ("prompt", "default prompt\n术语：MOSS\n热词提示：RTX 5070 Ti") in calls


async def test_moss_load_rejects_unsupported_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        raise AssertionError(f"unexpected import after dtype rejection: {name}")

    monkeypatch.setattr("asr_server.adapters.moss.importlib.import_module", fake_import_module)
    adapter = _MossWorkerBackend("moss-transcribe-diarize-0.9b")

    with pytest.raises(AsrError) as exc_info:
        await adapter.load("transformers", "cuda", "float32")

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "capability_not_supported"


async def test_moss_adapter_transcribe_uses_worker_protocol() -> None:
    adapter = MossTranscribeDiarizeAdapter("moss-transcribe-diarize-0.9b")
    conn = FakeConnection()
    adapter._transport = FakeTransport(conn)  # type: ignore[assignment]

    result = await adapter.transcribe(
        b"audio",
        model_id="moss-transcribe-diarize-0.9b",
        backend="transformers",
        language="zh",
        context="ctx",
        max_new_tokens=2048,
    )

    assert result.text == "worker text"
    assert result.timings.inference_ms == 8.0
    assert result.segments[0].speaker == "S01"
    assert conn.sent == [
        {
            "id": 1,
            "op": "transcribe",
            "audio": b"audio",
            "language": "zh",
            "context": "ctx",
            "max_new_tokens": 2048,
        }
    ]


@pytest.mark.parametrize(
    ("start", "end", "warning"),
    [
        (float("nan"), 1.0, "moss_segment_non_finite"),
        (0.0, float("inf"), "moss_segment_non_finite"),
        (-0.1, 0.5, "moss_segment_invalid_range"),
        (0.8, 0.2, "moss_segment_invalid_range"),
        (0.0, 2.0, "moss_segment_out_of_chunk_bounds"),
    ],
)
def test_moss_segment_invariants_drop_invalid_ranges(start: float, end: float, warning: str) -> None:
    backend = _MossWorkerBackend("moss-transcribe-diarize-0.9b")

    normalized, actual_warning = backend._normalize_segment(
        TranscriptionSegment(start=start, end=end, speaker="S01", text="hello"),
        chunk_duration=1.0,
    )

    assert normalized is None
    assert actual_warning == warning


def test_moss_parser_failure_preserves_raw_transcription_as_warning() -> None:
    backend = _MossWorkerBackend("moss-transcribe-diarize-0.9b")

    def fail(_text: str) -> object:
        raise ValueError("bad parse")

    helpers = _MossHelpers("prompt", object(), object(), fail)
    segments, warnings = backend._parse_segments("raw text", helpers, chunk_duration=1.0)

    assert segments == []
    assert warnings == ["moss_segment_parser_failed:ValueError"]
