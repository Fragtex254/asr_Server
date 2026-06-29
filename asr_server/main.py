from __future__ import annotations

import json
import socket
import importlib
from dataclasses import replace
from typing import Annotated, Any, cast

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from asr_server import __version__
from asr_server.adapters.base import AsrAdapter
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.adapters.qwen import QwenAsrAdapter
from asr_server.audio.merger import merge_transcription_results
from asr_server.audio.preprocess import normalize_audio_to_wav
from asr_server.audio.splitter import split_audio
from asr_server.config import Settings, load_settings
from asr_server.errors import AsrError, asr_error_handler, validation_error_handler
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import Backend, default_models


MAX_CONTEXT_CHARS = 4000
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS = 4096


class LoadRequest(BaseModel):
    backend: Backend = "auto"
    device: str = "cuda"
    dtype: str = "auto"
    max_new_tokens: int | None = None


class UnloadRequest(BaseModel):
    mode: str = "after_current_requests"
    reject_new_requests: bool = True
    cuda_empty_cache: bool = True


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


def build_adapter_context(context: str, hotwords: str | None) -> str:
    parts = []
    stripped_context = context.strip()
    if stripped_context:
        parts.append(stripped_context)
    normalized_hotwords = parse_hotwords(hotwords)
    if normalized_hotwords:
        parts.append("Hotwords: " + ", ".join(normalized_hotwords))
    adapter_context = "\n".join(parts)
    if len(adapter_context) > MAX_CONTEXT_CHARS:
        raise AsrError(
            400,
            "bad_request",
            "context exceeds the server context length limit",
            {"context_chars": len(adapter_context), "max_context_chars": MAX_CONTEXT_CHARS},
        )
    return adapter_context


def parse_hotwords(hotwords: str | None) -> list[str]:
    if hotwords is None or not hotwords.strip():
        return []
    stripped = hotwords.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AsrError(400, "bad_request", "hotwords must be a JSON string array or comma-separated string") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise AsrError(400, "bad_request", "hotwords JSON value must be an array of strings")
        return [item.strip() for item in parsed if item.strip()]
    return [item.strip() for item in stripped.split(",") if item.strip()]


def validate_max_new_tokens(max_new_tokens: int | None) -> int | None:
    if max_new_tokens is None:
        return DEFAULT_MAX_NEW_TOKENS
    if max_new_tokens > MAX_NEW_TOKENS:
        raise AsrError(
            400,
            "bad_request",
            "max_new_tokens exceeds the server limit",
            {"max_new_tokens": max_new_tokens, "max_new_tokens_limit": MAX_NEW_TOKENS},
        )
    return max_new_tokens


def create_app(settings: Settings | None = None, adapter_delay_seconds: float = 0.0) -> FastAPI:
    app_settings = settings or load_settings()

    def adapter_factory(model_id: str) -> AsrAdapter:
        if app_settings.adapter == "qwen":
            return QwenAsrAdapter(model_id)
        return MockAsrAdapter(delay_seconds=adapter_delay_seconds)

    app = FastAPI(title="WSL ASR Server", version=__version__)
    app.add_exception_handler(AsrError, cast(Any, asr_error_handler))
    app.add_exception_handler(RequestValidationError, cast(Any, validation_error_handler))
    app.state.manager = ModelLifecycleManager(default_models(app_settings.default_model), adapter_factory)
    app.state.settings = app_settings

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
    ) -> dict[str, object] | PlainTextResponse:
        del temperature
        if response_format not in {"json", "text", "verbose_json"}:
            raise AsrError(400, "bad_request", f"unsupported response_format: {response_format}")
        if timestamps not in {"none", "word", "char"}:
            raise AsrError(400, "bad_request", f"unsupported timestamps value: {timestamps}")
        manager: ModelLifecycleManager = app.state.manager
        selected_model = model or manager.default_model_id
        runtime = manager.runtime_for(selected_model)
        manager.resolve_backend(runtime, backend)
        if timestamps != "none" and not runtime.definition.capabilities.timestamps:
            raise AsrError(
                422,
                "capability_not_supported",
                f"{selected_model} does not support timestamps in this server",
                {"timestamps": timestamps},
            )
        adapter_context = build_adapter_context(context, hotwords)
        adapter_max_new_tokens = validate_max_new_tokens(max_new_tokens)
        audio = await file.read()
        normalized = normalize_audio_to_wav(audio)
        split = split_audio(
            normalized.audio,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        resolved_backend, chunk_results, timings = await manager.transcribe_chunks(
            [chunk.audio for chunk in split.chunks],
            model_id=model,
            backend=backend,
            language=language,
            timestamps=timestamps,
            context=adapter_context,
            max_new_tokens=adapter_max_new_tokens,
            batch_size=app_settings.qwen_batch_size,
        )
        timings = replace(
            timings,
            total_ms=timings.total_ms + normalized.decode_ms,
            decode_ms=timings.decode_ms + normalized.decode_ms,
        )
        result = merge_transcription_results(
            split.chunks,
            chunk_results,
            source_duration=split.metadata.duration_seconds,
            preserve_segments=preserve_segments,
            timings=timings,
        )
        generation_warnings = [f"max_new_tokens_override:{adapter_max_new_tokens}"] if max_new_tokens is not None else []
        warnings = list(dict.fromkeys([*result.warnings, *split.warnings, *generation_warnings]))
        if response_format == "text":
            return PlainTextResponse(result.text)
        return {
            "id": "tr_mock",
            "model": selected_model,
            "backend": resolved_backend,
            "language": result.language,
            "text": result.text,
            "duration": result.duration,
            "timestamps": [],
            "segments": [],
            "split": split.summary(),
            "chunks": result.chunks,
            "usage": {"audio_seconds": result.duration},
            "timings": result.timings.to_api(),
            "warnings": warnings,
        }

    @app.post("/v1/audio/alignments")
    async def alignments() -> dict[str, object]:
        raise AsrError(
            422,
            "capability_not_supported",
            "forced alignment is not enabled in this server",
        )

    return app


app = create_app()
