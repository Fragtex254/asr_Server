from __future__ import annotations

import asyncio
import io
import wave
from array import array
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server import main as main_module
from asr_server.adapters.base import TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.errors import AsrError
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


async def create_job(
    client: AsyncClient,
    *,
    audio: bytes | None = None,
    data: dict[str, str] | None = None,
) -> dict[str, object]:
    response = await client.post(
        "/v1/audio/transcription-jobs",
        files={"file": ("sample.wav", audio or make_wav(0.1), "audio/wav")},
        data=data or {},
    )
    assert response.status_code == 202
    return cast(dict[str, object], response.json())


async def wait_for_status(client: AsyncClient, job_id: str, statuses: set[str]) -> dict[str, object]:
    for _ in range(80):
        response = await client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        body = cast(dict[str, object], body)
        if body["status"] in statuses:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {statuses}")


async def test_create_job_returns_accepted_and_status_url(client: AsyncClient) -> None:
    body = await create_job(client)

    assert body["status"] == "queued"
    assert body["model"] == "qwen3-asr-1.7b"
    assert body["backend"] == "transformers"
    assert body["queue_position"] == 1
    assert body["status_url"] == f"/v1/jobs/{body['id']}"

    completed = await wait_for_status(client, str(body["id"]), {"completed"})
    completed_result = cast(dict[str, Any], completed["result"])
    completed_progress = cast(dict[str, Any], completed["progress"])
    assert completed_result["text"]
    assert completed_result["model"] == "qwen3-asr-1.7b"
    assert completed_progress["percent"] == 100.0


async def test_multiple_jobs_are_serialized_and_show_queue_position() -> None:
    app = create_app(adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)
        await wait_for_status(client, str(first["id"]), {"preprocessing", "splitting", "loading_model", "transcribing"})
        second = await create_job(client)

        queued = await client.get(f"/v1/jobs/{second['id']}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        assert queued.json()["queue_position"] == 1

        await wait_for_status(client, str(first["id"]), {"completed"})
        await wait_for_status(client, str(second["id"]), {"completed"})


async def test_job_reports_chunk_progress() -> None:
    app = create_app(adapter_delay_seconds=0.03)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(
            client,
            audio=make_wav(0.11),
            data={
                "split_strategy": "fixed",
                "max_chunk_seconds": "0.04",
                "overlap_seconds": "0.01",
                "preserve_segments": "true",
            },
        )

        transcribing = await wait_for_status(client, str(created["id"]), {"transcribing"})
        progress = cast(dict[str, Any], transcribing["progress"])
        assert progress["total_chunks"] == 4
        assert progress["current_chunk"] >= 1
        assert progress["completed_chunks"] >= 0

        completed = await wait_for_status(client, str(created["id"]), {"completed"})
        completed_progress = cast(dict[str, Any], completed["progress"])
        completed_result = cast(dict[str, Any], completed["result"])
        assert completed_progress["total_chunks"] == 4
        assert completed_progress["completed_chunks"] == 4
        assert len(cast(list[dict[str, object]], completed_result["chunks"])) == 4


async def test_job_failure_uses_error_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_transcribe(
        self: MockAsrAdapter,
        audio: bytes,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> TranscriptionResult:
        del self, audio, model_id, backend, language, context, max_new_tokens
        raise AsrError(503, "gpu_unavailable", "CUDA out of memory during Qwen ASR", {"phase": "transcribing"})

    monkeypatch.setattr(MockAsrAdapter, "transcribe", fail_transcribe)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(client)
        failed = await wait_for_status(client, str(created["id"]), {"failed"})

    assert failed["result"] is None
    error = cast(dict[str, Any], failed["error"])
    details = cast(dict[str, Any], error["details"])
    assert error["code"] == "gpu_unavailable"
    assert details["phase"] == "transcribing"


async def test_queued_job_can_be_cancelled() -> None:
    app = create_app(adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)
        second = await create_job(client)

        response = await client.delete(f"/v1/jobs/{second['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

        cancelled = await client.get(f"/v1/jobs/{second['id']}")
        assert cancelled.json()["status"] == "cancelled"
        await wait_for_status(client, str(first["id"]), {"completed"})


async def test_running_job_cancel_waits_for_chunk_boundary() -> None:
    app = create_app(adapter_delay_seconds=0.04)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(
            client,
            audio=make_wav(0.11),
            data={"split_strategy": "fixed", "max_chunk_seconds": "0.04", "overlap_seconds": "0.01"},
        )
        await wait_for_status(client, str(created["id"]), {"transcribing"})

        response = await client.delete(f"/v1/jobs/{created['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancel_requested"

        cancelled = await wait_for_status(client, str(created["id"]), {"cancelled"})
        assert cancelled["status"] == "cancelled"
        progress = cast(dict[str, Any], cancelled["progress"])
        assert progress["phase"] == "cancelled"


async def test_unknown_job_returns_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/v1/jobs/job_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


async def test_sync_transcription_over_duration_threshold_returns_job(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "SYNC_JOB_THRESHOLD_SECONDS", 0.01)

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/v1/jobs/{body['id']}"
