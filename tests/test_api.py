from __future__ import annotations

import asyncio
import io
import wave
from array import array
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server import main as main_module
from asr_server.main import create_app


def make_wav(duration_seconds: float, amplitude: int = 10_000, sample_rate: int = 16_000) -> bytes:
    samples = array("h", [amplitude] * int(duration_seconds * sample_rate))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_models_lists_only_qwen_models(client: AsyncClient) -> None:
    response = await client.get("/v1/models")

    assert response.status_code == 200
    models = response.json()["models"]
    model_ids = {model["id"] for model in models}
    assert model_ids == {"qwen3-asr-1.7b", "qwen3-asr-0.6b"}
    for model in models:
        assert model["capabilities"]["backends"] == ["transformers"]
        assert model["capabilities"]["streaming"] is False
        assert model["capabilities"]["timestamps"] == []
        assert model["capabilities"]["forced_alignment"] is False


async def test_unknown_model_uses_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/v1/models/missing/status")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_load_and_unload_model(client: AsyncClient) -> None:
    load_response = await client.post(
        "/v1/models/qwen3-asr-1.7b/load",
        json={"backend": "transformers", "device": "cuda", "dtype": "auto"},
    )
    assert load_response.status_code == 200
    assert load_response.json()["status"] == "loaded"

    status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
    assert status_response.json()["status"] == "loaded"
    assert status_response.json()["backend"] == "transformers"

    unload_response = await client.request(
        "DELETE",
        "/v1/models/qwen3-asr-1.7b",
        json={"mode": "after_current_requests", "reject_new_requests": True, "cuda_empty_cache": True},
    )
    assert unload_response.status_code == 200
    assert unload_response.json()["status"] == "unloaded"


async def test_loading_vllm_backend_is_not_supported(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/models/qwen3-asr-1.7b/load",
        json={"backend": "vllm", "device": "cuda", "dtype": "auto"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"


async def test_transcription_auto_loads_default_model(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"language": "auto", "backend": "transformers"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "qwen3-asr-1.7b"
    assert body["backend"] == "transformers"
    assert body["text"]
    assert body["timings"]["total_ms"] >= 0
    assert body["timings"]["decode_ms"] >= 0
    assert body["timings"]["load_ms"] >= 0
    assert body["timings"]["inference_ms"] >= 0
    assert body["usage"]["audio_seconds"] == body["duration"]


async def test_transcription_reports_zero_load_time_when_model_is_already_loaded(client: AsyncClient) -> None:
    load_response = await client.post(
        "/v1/models/qwen3-asr-1.7b/load",
        json={"backend": "transformers", "device": "cuda", "dtype": "auto"},
    )
    assert load_response.status_code == 200

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "language": "auto", "backend": "transformers"},
    )

    assert response.status_code == 200
    timings = response.json()["timings"]
    assert timings["total_ms"] >= 0
    assert timings["load_ms"] == 0


async def test_transcription_supports_each_declared_backend(client: AsyncClient) -> None:
    for model in ("qwen3-asr-0.6b", "qwen3-asr-1.7b"):
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": model, "backend": "transformers"},
        )

        assert response.status_code == 200
        assert response.json()["model"] == model
        assert response.json()["backend"] == "transformers"
        assert response.json()["text"]


async def test_transcription_can_return_chunk_metadata(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
        data={
            "model": "qwen3-asr-1.7b",
            "backend": "transformers",
            "split_strategy": "fixed",
            "max_chunk_seconds": "0.04",
            "overlap_seconds": "0.01",
            "preserve_segments": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["split"]["strategy"] == "fixed"
    assert body["split"]["chunk_count"] == 4
    assert len(body["chunks"]) == 4
    assert body["chunks"][0]["index"] == 0
    assert body["chunks"][1]["start"] == pytest.approx(0.03)
    assert body["chunks"][1]["raw_text"]
    assert body["chunks"][1]["deduped_prefix_chars"] >= 0
    assert body["chunks"][1]["timestamp_source"] == "vad_chunk_window"
    assert body["duration"] == pytest.approx(0.11)
    assert body["text"]


async def test_transcription_uses_vad_by_default(client: AsyncClient) -> None:
    samples = array("h")
    samples.extend([0] * int(0.2 * 16_000))
    samples.extend([10_000] * int(0.4 * 16_000))
    samples.extend([0] * int(0.5 * 16_000))
    samples.extend([10_000] * int(0.4 * 16_000))
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples.tobytes())

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", audio.getvalue(), "audio/wav")},
        data={
            "model": "qwen3-asr-1.7b",
            "backend": "transformers",
            "max_chunk_seconds": "0.8",
            "overlap_seconds": "0.03",
            "preserve_segments": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["split"]["strategy"] == "vad"
    assert body["split"]["requested_strategy"] == "auto"
    assert body["split"]["chunk_count"] == 2
    assert len(body["chunks"]) == 2
    assert body["duration"] == pytest.approx(1.5)


async def test_transcription_rejects_invalid_split_overlap(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
        data={
            "split_strategy": "fixed",
            "max_chunk_seconds": "0.04",
            "overlap_seconds": "0.04",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


async def test_vllm_backend_is_not_declared_for_first_release(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "backend": "vllm"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"


async def test_unsupported_capability_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "timestamps": "word"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"


async def test_unsupported_capability_is_rejected_before_audio_decode(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_decode(audio: bytes) -> object:
        nonlocal called
        called = True
        raise AssertionError("decode should not run for unsupported capabilities")

    monkeypatch.setattr(main_module, "normalize_audio_to_wav", fail_decode)

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"not actually audio", "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "timestamps": "word"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"
    assert called is False


async def test_unload_waits_for_active_request_and_rejects_new_requests() -> None:
    app = create_app(adapter_delay_seconds=0.05)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        transcription_task = asyncio.create_task(
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
                data={"model": "qwen3-asr-1.7b", "backend": "transformers"},
            )
        )
        await asyncio.sleep(0.01)

        unload_response = await client.request(
            "DELETE",
            "/v1/models/qwen3-asr-1.7b",
            json={"mode": "after_current_requests", "reject_new_requests": True, "cuda_empty_cache": True},
        )
        assert unload_response.status_code == 200
        assert unload_response.json()["status"] == "unloading_scheduled"
        assert unload_response.json()["active_requests"] == 1

        rejected_response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": "qwen3-asr-1.7b", "backend": "transformers"},
        )
        assert rejected_response.status_code == 409
        assert rejected_response.json()["error"]["code"] == "model_unloading_scheduled"

        completed_response = await transcription_task
        assert completed_response.status_code == 200

        status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
        assert status_response.json()["status"] == "unloaded"
        assert status_response.json()["active_requests"] == 0
