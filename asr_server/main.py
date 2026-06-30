from __future__ import annotations

import asyncio
import socket
import importlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from asr_server import __version__
from asr_server.adapters.base import AsrAdapter
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.adapters.qwen import QwenAsrAdapter
from asr_server.audio.metadata import inspect_audio
from asr_server.audio.preprocess import normalize_audio_to_wav, probe_audio_duration_seconds
from asr_server.config import Settings, load_settings
from asr_server.errors import AsrError, asr_error_handler, validation_error_handler
from asr_server.jobs import JobManager
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import Backend, default_models
from asr_server.transcription import (
    SYNC_JOB_THRESHOLD_SECONDS,
    TranscriptionRequest,
    run_transcription,
    validate_max_new_tokens,
    validate_transcription_request,
)


class LoadRequest(BaseModel):
    backend: Backend = "auto"
    device: str = "cuda"
    dtype: str = "auto"
    max_new_tokens: int | None = None


class UnloadRequest(BaseModel):
    mode: str = "after_current_requests"
    reject_new_requests: bool = True
    cuda_empty_cache: bool = True


UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


async def read_upload_limited(file: UploadFile, settings: Settings) -> bytes:
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_upload_bytes:
            raise AsrError(
                413,
                "audio_too_large",
                "audio upload exceeds the server size limit",
                {
                    "size_bytes": total_bytes,
                    "max_upload_bytes": max_upload_bytes,
                    "max_upload_mb": settings.max_upload_mb,
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


def gpu_health() -> dict[str, object]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return {"available": False, "name": None, "vram_total_mb": None}
    if getattr(torch.version, "cuda", None) is None or not torch.cuda.is_available():
        return {"available": False, "name": None, "vram_total_mb": None}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_total_mb": int(properties.total_memory / 1024 / 1024),
    }


def create_app(settings: Settings | None = None, adapter_delay_seconds: float = 0.0) -> FastAPI:
    app_settings = settings or load_settings()

    def adapter_factory(model_id: str) -> AsrAdapter:
        if app_settings.adapter == "qwen":
            return QwenAsrAdapter(model_id)
        return MockAsrAdapter(delay_seconds=adapter_delay_seconds)

    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI) -> AsyncIterator[None]:
        job_manager: JobManager = lifespan_app.state.job_manager
        manager: ModelLifecycleManager = lifespan_app.state.manager
        await job_manager.start()
        try:
            yield
        finally:
            await job_manager.shutdown()
            await manager.shutdown()

    app = FastAPI(title="WSL ASR Server", version=__version__, lifespan=lifespan)
    app.add_exception_handler(AsrError, cast(Any, asr_error_handler))
    app.add_exception_handler(RequestValidationError, cast(Any, validation_error_handler))
    app.state.manager = ModelLifecycleManager(
        default_models(app_settings.default_model),
        adapter_factory,
        idle_unload_seconds=app_settings.idle_unload_seconds,
    )
    app.state.settings = app_settings
    app.state.job_manager = JobManager(app.state.manager, app_settings)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "host": socket.gethostname(),
            "gpu": gpu_health(),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        manager: ModelLifecycleManager = app.state.manager
        return {"models": manager.list_models()}

    @app.get("/v1/models/{model_id}/status")
    async def model_status(model_id: str) -> dict[str, object]:
        manager: ModelLifecycleManager = app.state.manager
        return manager.runtime_for(model_id).status_summary()

    @app.post("/v1/models/{model_id}/load")
    async def load_model(
        model_id: str,
        request: Annotated[LoadRequest, Body()] = LoadRequest(),
    ) -> dict[str, object]:
        manager: ModelLifecycleManager = app.state.manager
        return await manager.load_model(
            model_id,
            backend=request.backend,
            device=request.device,
            dtype=request.dtype,
            max_new_tokens=validate_max_new_tokens(request.max_new_tokens),
        )

    @app.delete("/v1/models/{model_id}")
    async def unload_model(
        model_id: str,
        request: Annotated[UnloadRequest, Body()] = UnloadRequest(),
    ) -> dict[str, object]:
        del request.mode, request.reject_new_requests
        manager: ModelLifecycleManager = app.state.manager
        return await manager.unload_model(model_id, cuda_empty_cache=request.cuda_empty_cache)

    @app.delete("/v1/models")
    async def unload_models(request: Annotated[UnloadRequest, Body()] = UnloadRequest()) -> dict[str, object]:
        del request.mode, request.reject_new_requests
        manager: ModelLifecycleManager = app.state.manager
        return await manager.unload_all(cuda_empty_cache=request.cuda_empty_cache)

    @app.post("/v1/audio/transcriptions", response_model=None)
    async def transcribe(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str, Form()] = "auto",
        response_format: Annotated[str, Form()] = "json",
        timestamps: Annotated[str, Form()] = "none",
        backend: Annotated[Backend, Form()] = "auto",
        temperature: Annotated[float | None, Form()] = None,
        max_new_tokens: Annotated[int | None, Form(ge=1)] = None,
        context: Annotated[str, Form()] = "",
        hotwords: Annotated[str | None, Form()] = None,
        split_strategy: Annotated[str, Form()] = "auto",
        max_chunk_seconds: Annotated[float | None, Form()] = None,
        overlap_seconds: Annotated[float | None, Form()] = None,
        preserve_segments: Annotated[bool, Form()] = False,
    ) -> dict[str, object] | PlainTextResponse | JSONResponse:
        del temperature
        manager: ModelLifecycleManager = app.state.manager
        request = _transcription_request(
            model=model,
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            backend=backend,
            max_new_tokens=max_new_tokens,
            context=context,
            hotwords=hotwords,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
            preserve_segments=preserve_segments,
        )
        validated = validate_transcription_request(manager, request)
        audio = await read_upload_limited(file, app_settings)
        duration_seconds = await asyncio.to_thread(probe_audio_duration_seconds, audio)
        if duration_seconds is None:
            normalized = await asyncio.to_thread(normalize_audio_to_wav, audio)
            metadata = await asyncio.to_thread(inspect_audio, normalized.audio)
            duration_seconds = metadata.duration_seconds
        if duration_seconds > SYNC_JOB_THRESHOLD_SECONDS:
            job_manager: JobManager = app.state.job_manager
            payload = await job_manager.create_job(
                audio=audio,
                filename=file.filename,
                content_type=file.content_type,
                request=request,
            )
            return JSONResponse(status_code=202, content=payload)
        result = await run_transcription(
            audio,
            manager=manager,
            settings=app_settings,
            request=request,
            validated=validated,
        )
        if response_format == "text":
            return PlainTextResponse(cast(str, result["text"]))
        return result

    @app.post("/v1/audio/transcription-jobs", status_code=202)
    async def create_transcription_job(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str, Form()] = "auto",
        response_format: Annotated[str, Form()] = "json",
        timestamps: Annotated[str, Form()] = "none",
        backend: Annotated[Backend, Form()] = "auto",
        temperature: Annotated[float | None, Form()] = None,
        max_new_tokens: Annotated[int | None, Form(ge=1)] = None,
        context: Annotated[str, Form()] = "",
        hotwords: Annotated[str | None, Form()] = None,
        split_strategy: Annotated[str, Form()] = "auto",
        max_chunk_seconds: Annotated[float | None, Form()] = None,
        overlap_seconds: Annotated[float | None, Form()] = None,
        preserve_segments: Annotated[bool, Form()] = False,
    ) -> dict[str, object]:
        del temperature
        request = _transcription_request(
            model=model,
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            backend=backend,
            max_new_tokens=max_new_tokens,
            context=context,
            hotwords=hotwords,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
            preserve_segments=preserve_segments,
        )
        audio = await read_upload_limited(file, app_settings)
        job_manager: JobManager = app.state.job_manager
        return await job_manager.create_job(
            audio=audio,
            filename=file.filename,
            content_type=file.content_type,
            request=request,
        )

    @app.get("/v1/jobs/{job_id}")
    async def get_transcription_job(job_id: str) -> dict[str, object]:
        job_manager: JobManager = app.state.job_manager
        return await job_manager.get_job(job_id)

    @app.delete("/v1/jobs/{job_id}")
    async def cancel_transcription_job(job_id: str) -> dict[str, object]:
        job_manager: JobManager = app.state.job_manager
        return await job_manager.cancel_job(job_id)

    @app.post("/v1/audio/alignments")
    async def alignments() -> dict[str, object]:
        raise AsrError(
            422,
            "capability_not_supported",
            "forced alignment is not enabled in this server",
        )

    return app


def _transcription_request(
    *,
    model: str | None,
    language: str,
    response_format: str,
    timestamps: str,
    backend: Backend,
    max_new_tokens: int | None,
    context: str,
    hotwords: str | None,
    split_strategy: str,
    max_chunk_seconds: float | None,
    overlap_seconds: float | None,
    preserve_segments: bool,
) -> TranscriptionRequest:
    return TranscriptionRequest(
        model=model,
        language=language,
        response_format=response_format,
        timestamps=timestamps,
        backend=backend,
        max_new_tokens=max_new_tokens,
        context=context,
        hotwords=hotwords,
        split_strategy=split_strategy,
        max_chunk_seconds=max_chunk_seconds,
        overlap_seconds=overlap_seconds,
        preserve_segments=preserve_segments,
    )


app = create_app()
