from __future__ import annotations

import json
import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from asr_server.adapters.base import TranscriptionResult
from asr_server.audio.merger import merge_transcription_results
from asr_server.audio.preprocess import normalize_audio_path_to_wav_async, normalize_audio_to_wav
from asr_server.audio.splitter import PathSplitResult, SplitCancelled, SplitResult, split_audio, split_audio_path
from asr_server.config import Settings
from asr_server.audio.workspace import validate_workspace_limits
from asr_server.errors import AsrError
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import MOSS_MODEL_ID, Backend


MAX_CONTEXT_CHARS = 4000
DEFAULT_MAX_NEW_TOKENS = 512
MOSS_DEFAULT_MAX_NEW_TOKENS = 2048
MAX_NEW_TOKENS = 4096
SYNC_JOB_THRESHOLD_SECONDS = 600.0

StageCallback = Callable[[str, dict[str, object]], Awaitable[None]]
BeforeChunkCallback = Callable[[int, int], Awaitable[None]]
AfterChunkCallback = Callable[[int, int, TranscriptionResult], Awaitable[None]]


@dataclass(frozen=True)
class GenerationConfig:
    language: str
    max_new_tokens: int | None
    context: str
    hotwords: str | None


@dataclass(frozen=True)
class RequestConfig:
    response_format: str
    timestamps: str
    split_strategy: str
    max_chunk_seconds: float | None
    overlap_seconds: float | None
    preserve_segments: bool


@dataclass(frozen=True)
class TranscriptionRequest:
    model: str | None
    language: str
    response_format: str
    timestamps: str
    backend: Backend
    max_new_tokens: int | None
    context: str
    hotwords: str | None
    split_strategy: str
    max_chunk_seconds: float | None
    overlap_seconds: float | None
    preserve_segments: bool

    @property
    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(self.language, self.max_new_tokens, self.context, self.hotwords)

    @property
    def request_config(self) -> RequestConfig:
        return RequestConfig(
            self.response_format,
            self.timestamps,
            self.split_strategy,
            self.max_chunk_seconds,
            self.overlap_seconds,
            self.preserve_segments,
        )


@dataclass(frozen=True)
class ValidatedTranscription:
    selected_model: str
    resolved_backend: str
    adapter_context: str
    max_new_tokens: int | None


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


def build_adapter_context(context: str, hotwords: str | None) -> str:
    parts = []
    stripped_context = context.strip()
    if stripped_context:
        parts.append(stripped_context)
    normalized_hotwords = parse_hotwords(hotwords)
    if normalized_hotwords:
        parts.append("热词提示：" + ", ".join(normalized_hotwords))
    adapter_context = "\n".join(parts)
    if len(adapter_context) > MAX_CONTEXT_CHARS:
        raise AsrError(
            400,
            "bad_request",
            "context exceeds the server context length limit",
            {"context_chars": len(adapter_context), "max_context_chars": MAX_CONTEXT_CHARS},
        )
    return adapter_context


def validate_max_new_tokens(
    max_new_tokens: int | None,
    *,
    default: int = DEFAULT_MAX_NEW_TOKENS,
) -> int | None:
    if max_new_tokens is None:
        return default
    if max_new_tokens > MAX_NEW_TOKENS:
        raise AsrError(
            400,
            "bad_request",
            "max_new_tokens exceeds the server limit",
            {"max_new_tokens": max_new_tokens, "max_new_tokens_limit": MAX_NEW_TOKENS},
        )
    return max_new_tokens


def default_max_new_tokens_for_model(model_id: str) -> int:
    return MOSS_DEFAULT_MAX_NEW_TOKENS if model_id == MOSS_MODEL_ID else DEFAULT_MAX_NEW_TOKENS


def validate_max_new_tokens_for_model(model_id: str, max_new_tokens: int | None) -> int | None:
    return validate_max_new_tokens(
        max_new_tokens,
        default=default_max_new_tokens_for_model(model_id),
    )


def validate_transcription_request(
    manager: ModelLifecycleManager,
    request: TranscriptionRequest,
) -> ValidatedTranscription:
    if request.response_format not in {"json", "text", "verbose_json"}:
        raise AsrError(400, "bad_request", f"unsupported response_format: {request.response_format}")
    if request.timestamps not in {"none", "word", "char"}:
        raise AsrError(400, "bad_request", f"unsupported timestamps value: {request.timestamps}")
    selected_model = request.model or manager.default_model_id
    runtime = manager.runtime_for(selected_model)
    resolved_backend = manager.resolve_backend(runtime, request.backend)
    if request.language not in runtime.definition.capabilities.languages:
        raise AsrError(
            422,
            "capability_not_supported",
            f"{selected_model} does not support language: {request.language}",
            {"language": request.language},
        )
    if request.timestamps != "none" and not runtime.definition.capabilities.timestamps:
        raise AsrError(
            422,
            "capability_not_supported",
            f"{selected_model} does not support timestamps in this server",
            {"timestamps": request.timestamps},
        )
    return ValidatedTranscription(
        selected_model=selected_model,
        resolved_backend=resolved_backend,
        adapter_context=build_adapter_context(request.context, request.hotwords),
        max_new_tokens=validate_max_new_tokens_for_model(selected_model, request.max_new_tokens),
    )


async def run_transcription(
    audio: bytes,
    *,
    manager: ModelLifecycleManager,
    settings: Settings,
    request: TranscriptionRequest,
    validated: ValidatedTranscription | None = None,
    stage_callback: StageCallback | None = None,
    before_chunk: BeforeChunkCallback | None = None,
    after_chunk: AfterChunkCallback | None = None,
) -> dict[str, object]:
    checked = validated or validate_transcription_request(manager, request)
    if stage_callback is not None:
        await stage_callback("preprocessing", {"percent": 1.0})
    normalized = await asyncio.to_thread(normalize_audio_to_wav, audio)
    if stage_callback is not None:
        await stage_callback("splitting", {"percent": 3.0})
    split = await asyncio.to_thread(
        lambda: split_audio(
            normalized.audio,
            split_strategy=request.split_strategy,
            max_chunk_seconds=request.max_chunk_seconds,
            overlap_seconds=request.overlap_seconds,
        )
    )
    if stage_callback is not None:
        await stage_callback(
            "loading_model",
            {
                "percent": 5.0,
                "split": split.summary(),
                "total_chunks": len(split.chunks),
                "completed_chunks": 0,
                "chunk_windows": [(chunk.start, chunk.end) for chunk in split.chunks],
            },
        )
    resolved_backend, chunk_results, timings = await manager.transcribe_chunks(
        [chunk.audio for chunk in split.chunks],
        model_id=request.model,
        backend=request.backend,
        language=request.language,
        timestamps=request.timestamps,
        context=checked.adapter_context,
        max_new_tokens=checked.max_new_tokens,
        batch_size=settings.qwen_batch_size,
        before_chunk=before_chunk,
        after_chunk=after_chunk,
    )
    timings = replace(
        timings,
        total_ms=timings.total_ms + normalized.decode_ms,
        decode_ms=timings.decode_ms + normalized.decode_ms,
    )
    if stage_callback is not None:
        await stage_callback("merging", {"percent": 99.0})
    result = await asyncio.to_thread(
        merge_transcription_results,
        split.chunks,
        chunk_results,
        source_duration=split.metadata.duration_seconds,
        preserve_segments=request.preserve_segments,
        timings=timings,
    )
    warnings = _warnings(result.warnings, split.warnings, checked.max_new_tokens, request.max_new_tokens)
    return transcription_payload(
        id_=f"tr_{uuid4().hex}",
        selected_model=checked.selected_model,
        resolved_backend=resolved_backend,
        split=split,
        text=result.text,
        language=result.language,
        duration=result.duration,
        chunks=result.chunks,
        segments=result.segments if request.response_format == "verbose_json" else [],
        timings=result.timings.to_api(),
        warnings=warnings,
    )


async def run_transcription_path(
    upload_path: Path,
    *,
    workspace: Path,
    manager: ModelLifecycleManager,
    settings: Settings,
    request: TranscriptionRequest,
    validated: ValidatedTranscription | None = None,
    stage_callback: StageCallback | None = None,
    before_chunk: BeforeChunkCallback | None = None,
    after_chunk: AfterChunkCallback | None = None,
) -> dict[str, object]:
    checked = validated or validate_transcription_request(manager, request)
    if stage_callback is not None:
        await stage_callback("preprocessing", {"percent": 1.0})
    normalized = await normalize_audio_path_to_wav_async(
        upload_path,
        workspace / "normalized.wav",
        timeout_seconds=settings.ffmpeg_timeout_seconds,
    )
    await asyncio.to_thread(validate_workspace_limits, workspace, settings)
    if stage_callback is not None:
        await stage_callback("splitting", {"percent": 3.0})
    split_cancel = threading.Event()
    split_operation = asyncio.create_task(
        asyncio.to_thread(
            split_audio_path,
            normalized.path,
            split_strategy=request.split_strategy,
            max_chunk_seconds=request.max_chunk_seconds,
            overlap_seconds=request.overlap_seconds,
            cancel_event=split_cancel,
        )
    )
    try:
        split = await asyncio.shield(split_operation)
    except asyncio.CancelledError:
        split_cancel.set()
        try:
            await split_operation
        except SplitCancelled:
            pass
        raise
    if stage_callback is not None:
        await stage_callback(
            "loading_model",
            {
                "percent": 5.0,
                "split": split.summary(),
                "total_chunks": len(split.chunks),
                "completed_chunks": 0,
                "chunk_windows": [(chunk.start, chunk.end) for chunk in split.chunks],
            },
        )
    resolved_backend, chunk_results, timings = await manager.transcribe_chunks(
        [chunk.as_audio_input() for chunk in split.chunks],
        model_id=request.model,
        backend=request.backend,
        language=request.language,
        timestamps=request.timestamps,
        context=checked.adapter_context,
        max_new_tokens=checked.max_new_tokens,
        batch_size=1,
        before_chunk=before_chunk,
        after_chunk=after_chunk,
    )
    await asyncio.to_thread(validate_workspace_limits, workspace, settings)
    timings = replace(
        timings,
        total_ms=timings.total_ms + normalized.decode_ms,
        decode_ms=timings.decode_ms + normalized.decode_ms,
    )
    if stage_callback is not None:
        await stage_callback("merging", {"percent": 99.0})
    result = await asyncio.to_thread(
        merge_transcription_results,
        split.chunks,
        chunk_results,
        source_duration=split.metadata.duration_seconds,
        preserve_segments=request.preserve_segments,
        timings=timings,
    )
    warnings = _warnings(result.warnings, split.warnings, checked.max_new_tokens, request.max_new_tokens)
    return transcription_payload(
        id_=f"tr_{uuid4().hex}",
        selected_model=checked.selected_model,
        resolved_backend=resolved_backend,
        split=split,
        text=result.text,
        language=result.language,
        duration=result.duration,
        chunks=result.chunks,
        segments=result.segments if request.response_format == "verbose_json" else [],
        timings=result.timings.to_api(),
        warnings=warnings,
    )


def transcription_payload(
    *,
    id_: str,
    selected_model: str,
    resolved_backend: str,
    split: SplitResult | PathSplitResult,
    text: str,
    language: str,
    duration: float,
    chunks: list[dict[str, object]],
    segments: list[dict[str, object]],
    timings: dict[str, float],
    warnings: list[str],
) -> dict[str, object]:
    return {
        "id": id_,
        "model": selected_model,
        "backend": resolved_backend,
        "language": language,
        "text": text,
        "duration": duration,
        "timestamps": [],
        "segments": segments,
        "split": split.summary(),
        "chunks": chunks,
        "usage": {"audio_seconds": duration},
        "timings": timings,
        "warnings": warnings,
    }


def _warnings(
    result_warnings: list[str],
    split_warnings: list[str],
    max_new_tokens: int | None,
    requested_max_new_tokens: int | None,
) -> list[str]:
    generation_warnings = [f"max_new_tokens_override:{max_new_tokens}"] if requested_max_new_tokens is not None else []
    return list(dict.fromkeys([*result_warnings, *split_warnings, *generation_warnings]))
