from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.main import create_app


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
        assert model["capabilities"]["backends"] == ["transformers", "vllm"]
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
        json={"backend": "vllm", "device": "cuda", "dtype": "auto"},
    )
    assert load_response.status_code == 200
    assert load_response.json()["status"] == "loaded"

    status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
    assert status_response.json()["status"] == "loaded"
    assert status_response.json()["backend"] == "vllm"

    unload_response = await client.request(
        "DELETE",
        "/v1/models/qwen3-asr-1.7b",
        json={"mode": "after_current_requests", "reject_new_requests": True, "cuda_empty_cache": True},
    )
    assert unload_response.status_code == 200
    assert unload_response.json()["status"] == "unloaded"


async def test_transcription_auto_loads_default_model(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"language": "auto", "backend": "transformers"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "qwen3-asr-1.7b"
    assert body["backend"] == "transformers"
    assert body["text"]


async def test_transcription_supports_each_declared_backend(client: AsyncClient) -> None:
    for model in ("qwen3-asr-0.6b", "qwen3-asr-1.7b"):
        for backend in ("transformers", "vllm"):
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"fake audio", "audio/wav")},
                data={"model": model, "backend": backend},
            )

            assert response.status_code == 200
            assert response.json()["model"] == model
            assert response.json()["backend"] == backend
            assert response.json()["text"]


async def test_unsupported_capability_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "timestamps": "word"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"


async def test_unload_waits_for_active_request_and_rejects_new_requests() -> None:
    app = create_app(adapter_delay_seconds=0.05)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        transcription_task = asyncio.create_task(
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"fake audio", "audio/wav")},
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
            files={"file": ("sample.wav", b"fake audio", "audio/wav")},
            data={"model": "qwen3-asr-1.7b", "backend": "transformers"},
        )
        assert rejected_response.status_code == 409
        assert rejected_response.json()["error"]["code"] == "model_unloading_scheduled"

        completed_response = await transcription_task
        assert completed_response.status_code == 200

        status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
        assert status_response.json()["status"] == "unloaded"
        assert status_response.json()["active_requests"] == 0

