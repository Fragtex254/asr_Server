from __future__ import annotations

import gc
import importlib
import logging
import multiprocessing
import signal
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from time import perf_counter
from typing import Any

from asr_server.adapters.base import TranscriptionResult, TranscriptionSegment, TranscriptionTimings
from asr_server.errors import AsrError
from asr_server.registry import MOSS_MODEL_ID


MODEL_REPOS = {
    MOSS_MODEL_ID: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
}
DEFAULT_MOSS_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
logger = logging.getLogger(__name__)


def _moss_worker_main(conn: Connection, model_id: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = _MossWorkerBackend(model_id)
    try:
        while True:
            request = conn.recv()
            request_id = request.get("id")
            op = request.get("op")
            try:
                if op == "load":
                    _run_async(
                        backend.load(
                            str(request["backend"]),
                            str(request["device"]),
                            str(request["dtype"]),
                            max_new_tokens=_optional_int(request.get("max_new_tokens")),
                        )
                    )
                    conn.send({"id": request_id, "ok": True, "result": None})
                    continue
                if op == "transcribe":
                    result = _run_async(
                        backend.transcribe(
                            bytes(request["audio"]),
                            model_id=model_id,
                            backend=backend.loaded_backend or "transformers",
                            language=str(request["language"]),
                            context=str(request["context"]),
                            max_new_tokens=_optional_int(request.get("max_new_tokens")),
                        )
                    )
                    conn.send({"id": request_id, "ok": True, "result": _result_to_payload(result)})
                    continue
                if op == "transcribe_batch":
                    results = _run_async(
                        backend.transcribe_batch(
                            list(request["audio_chunks"]),
                            model_id=model_id,
                            backend=backend.loaded_backend or "transformers",
                            language=str(request["language"]),
                            context=str(request["context"]),
                            max_new_tokens=_optional_int(request.get("max_new_tokens")),
                        )
                    )
                    conn.send({"id": request_id, "ok": True, "result": [_result_to_payload(item) for item in results]})
                    continue
                if op == "shutdown":
                    _run_async(backend.unload(cuda_empty_cache=bool(request.get("cuda_empty_cache", True))))
                    conn.send({"id": request_id, "ok": True, "result": None})
                    return
                raise AsrError(400, "bad_request", f"unknown MOSS worker operation: {op}")
            except AsrError as exc:
                conn.send({"id": request_id, "ok": False, "error": _error_to_payload(exc)})
            except Exception as exc:
                error = AsrError(
                    503,
                    "inference_failed",
                    "MOSS worker operation failed",
                    {"operation": str(op), "error_type": type(exc).__name__, "message": str(exc)[-500:]},
                )
                conn.send({"id": request_id, "ok": False, "error": _error_to_payload(error)})
    except EOFError:
        return
    finally:
        try:
            _run_async(backend.unload(cuda_empty_cache=True))
        finally:
            conn.close()


def _run_async(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected optional int value, got {type(value).__name__}")


def _result_to_payload(result: TranscriptionResult) -> dict[str, object]:
    return {
        "text": result.text,
        "duration": result.duration,
        "language": result.language,
        "warnings": result.warnings,
        "segments": [segment.to_api() for segment in result.segments],
        "timings": {
            "total_ms": result.timings.total_ms,
            "load_ms": result.timings.load_ms,
            "decode_ms": result.timings.decode_ms,
            "inference_ms": result.timings.inference_ms,
            "postprocess_ms": result.timings.postprocess_ms,
        },
    }


def _result_from_payload(payload: Any) -> TranscriptionResult:
    timings = payload.get("timings", {})
    return TranscriptionResult(
        text=str(payload["text"]),
        duration=float(payload["duration"]),
        language=str(payload["language"]),
        warnings=[str(warning) for warning in payload.get("warnings", [])],
        segments=[
            TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                speaker=str(segment["speaker"]) if segment.get("speaker") is not None else None,
                text=str(segment["text"]),
            )
            for segment in payload.get("segments", [])
        ],
        timings=TranscriptionTimings(
            total_ms=float(timings.get("total_ms", 0.0)),
            load_ms=float(timings.get("load_ms", 0.0)),
            decode_ms=float(timings.get("decode_ms", 0.0)),
            inference_ms=float(timings.get("inference_ms", 0.0)),
            postprocess_ms=float(timings.get("postprocess_ms", 0.0)),
        ),
    )


def _error_to_payload(error: AsrError) -> dict[str, object]:
    return {
        "status_code": error.status_code,
        "code": error.code,
        "message": error.message,
        "details": error.details,
    }


class MossTranscribeDiarizeAdapter:
    """MOSS-Transcribe-Diarize adapter with GPU work isolated in a child process."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.loaded_backend: str | None = None
        self._worker: Any | None = None
        self._conn: Connection | None = None
        self._request_id = 0

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        if self._worker is None or self._conn is None or not self._worker.is_alive():
            self._start_worker()
        self._request("load", backend=backend, device=device, dtype=dtype, max_new_tokens=max_new_tokens)
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        self.loaded_backend = None
        worker = self._worker
        conn = self._conn
        self._worker = None
        self._conn = None
        if worker is None:
            return
        if conn is not None and worker.is_alive():
            try:
                self._request_on(conn, "shutdown", cuda_empty_cache=cuda_empty_cache)
            except AsrError as exc:
                logger.warning("MOSS worker graceful shutdown failed: %s", exc.message)
            except (BrokenPipeError, EOFError, OSError) as exc:
                logger.warning("MOSS worker pipe closed during shutdown: %s", exc)
        if worker.is_alive():
            worker.join(timeout=10)
        if worker.is_alive():
            logger.warning("MOSS worker did not exit after unload; terminating pid=%s", worker.pid)
            worker.terminate()
            worker.join(timeout=5)
        if worker.is_alive():
            logger.warning("MOSS worker did not terminate cleanly; killing pid=%s", worker.pid)
            worker.kill()
            worker.join(timeout=5)
        if conn is not None:
            conn.close()

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
        del model_id, backend, language
        payload = self._request(
            "transcribe",
            audio=audio,
            language="auto",
            context=context,
            max_new_tokens=max_new_tokens,
        )
        return _result_from_payload(payload)

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
        del model_id, backend, language
        payload = self._request(
            "transcribe_batch",
            audio_chunks=audio_chunks,
            language="auto",
            context=context,
            max_new_tokens=max_new_tokens,
        )
        return [_result_from_payload(item) for item in payload]

    def _start_worker(self) -> None:
        parent_conn, child_conn = multiprocessing.get_context("spawn").Pipe()
        worker = multiprocessing.get_context("spawn").Process(
            target=_moss_worker_main,
            args=(child_conn, self.model_id),
            daemon=True,
        )
        worker.start()
        child_conn.close()
        self._worker = worker
        self._conn = parent_conn

    def _request(self, op: str, **payload: object) -> Any:
        conn = self._conn
        worker = self._worker
        if conn is None or worker is None or not worker.is_alive():
            raise AsrError(409, "model_loading", "MOSS worker is not running")
        return self._request_on(conn, op, **payload)

    def _request_on(self, conn: Connection, op: str, **payload: object) -> Any:
        self._request_id += 1
        request_id = self._request_id
        conn.send({"id": request_id, "op": op, **payload})
        response = conn.recv()
        if response.get("id") != request_id:
            raise AsrError(503, "inference_failed", "MOSS worker returned an unexpected response")
        if response.get("ok"):
            return response.get("result")
        error = response.get("error", {})
        raise AsrError(
            int(error.get("status_code", 503)),
            str(error.get("code", "inference_failed")),
            str(error.get("message", "MOSS worker failed")),
            dict(error.get("details", {})),
        )


@dataclass(frozen=True)
class _MossHelpers:
    default_prompt: str
    build_transcription_messages: Any
    generate_transcription: Any
    parse_transcript: Any


class _MossWorkerBackend:
    """In-process MOSS backend used only inside the worker process."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.loaded_backend: str | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._helpers: _MossHelpers | None = None
        self._max_new_tokens: int | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        if self.model_id not in MODEL_REPOS:
            raise AsrError(404, "model_not_found", f"unknown MOSS model: {self.model_id}")
        if device not in {"cuda", "cuda:0"}:
            raise AsrError(422, "capability_not_supported", "MOSS adapter requires device=cuda or cuda:0")
        if backend != "transformers":
            raise AsrError(422, "capability_not_supported", f"unsupported MOSS backend: {backend}")
        torch = self._assert_cuda_torch()
        torch_dtype = self._resolve_torch_dtype(torch, dtype)
        torch_device = torch.device(device)
        processor_cls, model_cls = self._import_transformers()
        helpers = self._import_moss_helpers()
        repo_id = MODEL_REPOS[self.model_id]
        if self._model is not None:
            await self.unload(cuda_empty_cache=True)
        try:
            model = (
                model_cls.from_pretrained(repo_id, trust_remote_code=True, dtype="auto")
                .to(dtype=torch_dtype)
                .to(torch_device)
                .eval()
            )
            processor = processor_cls.from_pretrained(repo_id, trust_remote_code=True)
            self._model = model
            self._processor = processor
            self._torch = torch
            self._device = torch_device
            self._dtype = torch_dtype
            self._helpers = helpers
            self._max_new_tokens = max_new_tokens
        except AsrError:
            raise
        except Exception as exc:
            raise self._map_moss_exception(exc, phase="load", model_id=repo_id) from exc
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        model = self._model
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._helpers = None
        self._max_new_tokens = None
        self.loaded_backend = None
        if model is not None:
            self._move_model_to_cpu(model)
            self._close_model(model)
            del model
        gc.collect()
        if cuda_empty_cache:
            self._release_cuda_cache()

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
        del model_id, backend, language
        model = self._model
        processor = self._processor
        helpers = self._helpers
        if model is None or processor is None or helpers is None:
            raise AsrError(409, "model_loading", "MOSS model is not loaded")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(audio)
            audio_path = Path(audio_file.name)
        try:
            inference_started = perf_counter()
            try:
                result = helpers.generate_transcription(
                    model,
                    processor,
                    helpers.build_transcription_messages(
                        str(audio_path),
                        prompt=self._prompt_for_context(context, helpers.default_prompt),
                    ),
                    max_new_tokens=max_new_tokens or self._max_new_tokens or 2048,
                    do_sample=False,
                    device=self._device,
                    dtype=self._dtype,
                )
            except Exception as exc:
                raise self._map_moss_exception(exc, phase="inference", model_id=self.model_id) from exc
            inference_ms = (perf_counter() - inference_started) * 1000
            postprocess_started = perf_counter()
            raw_text = self._result_text(result)
            if not raw_text.strip():
                raise AsrError(503, "inference_failed", "MOSS returned an empty transcription result")
            segments = self._parse_segments(raw_text, helpers)
            text = "\n".join(segment.text for segment in segments) if segments else raw_text
            postprocess_ms = (perf_counter() - postprocess_started) * 1000
            return TranscriptionResult(
                text=text,
                duration=max(len(audio) / 16_000, 0.01),
                language="auto",
                warnings=[],
                segments=segments,
                timings=TranscriptionTimings(
                    inference_ms=inference_ms,
                    postprocess_ms=postprocess_ms,
                ),
            )
        finally:
            audio_path.unlink(missing_ok=True)

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
        results = []
        for audio in audio_chunks:
            results.append(
                await self.transcribe(
                    audio,
                    model_id=model_id,
                    backend=backend,
                    language=language,
                    context=context,
                    max_new_tokens=max_new_tokens,
                )
            )
        return results

    def _assert_cuda_torch(self) -> Any:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise AsrError(503, "gpu_unavailable", "torch is not installed") from exc
        if torch.version.cuda is None:
            raise AsrError(503, "gpu_unavailable", "installed torch is CPU-only")
        if not torch.cuda.is_available():
            raise AsrError(503, "gpu_unavailable", "torch cannot access CUDA")
        return torch

    def _resolve_torch_dtype(self, torch: Any, dtype: str) -> Any:
        normalized = dtype.lower()
        if normalized == "auto" or normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16"}:
            return torch.float16
        raise AsrError(
            422,
            "capability_not_supported",
            "MOSS adapter supports dtype=auto, bfloat16/bf16, or float16/fp16",
            {"dtype": dtype},
        )

    def _import_transformers(self) -> tuple[Any, Any]:
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "transformers is not installed; install MOSS dependencies in WSL after CUDA torch validation",
            ) from exc
        processor_cls = getattr(transformers, "AutoProcessor", None)
        model_cls = getattr(transformers, "AutoModelForCausalLM", None)
        if processor_cls is None or model_cls is None:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "installed transformers does not provide MOSS loading classes",
                {
                    "missing": [
                        name
                        for name, value in (
                            ("AutoProcessor", processor_cls),
                            ("AutoModelForCausalLM", model_cls),
                        )
                        if value is None
                    ]
                },
            )
        return processor_cls, model_cls

    def _import_moss_helpers(self) -> _MossHelpers:
        try:
            moss_package = importlib.import_module("moss_transcribe_diarize")
            inference_utils = importlib.import_module("moss_transcribe_diarize.inference_utils")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "moss-transcribe-diarize is not installed in this WSL environment",
            ) from exc
        parse_transcript = getattr(moss_package, "parse_transcript", None)
        build_messages = getattr(inference_utils, "build_transcription_messages", None)
        generate_transcription = getattr(inference_utils, "generate_transcription", None)
        if not callable(parse_transcript) or not callable(build_messages) or not callable(generate_transcription):
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "installed moss-transcribe-diarize package is missing required inference helpers",
                {
                    "missing": [
                        name
                        for name, value in (
                            ("parse_transcript", parse_transcript),
                            ("build_transcription_messages", build_messages),
                            ("generate_transcription", generate_transcription),
                        )
                        if not callable(value)
                    ]
                },
            )
        return _MossHelpers(
            default_prompt=str(getattr(inference_utils, "DEFAULT_PROMPT", DEFAULT_MOSS_PROMPT)),
            build_transcription_messages=build_messages,
            generate_transcription=generate_transcription,
            parse_transcript=parse_transcript,
        )

    def _prompt_for_context(self, context: str, default_prompt: str) -> str:
        stripped = context.strip()
        if not stripped:
            return default_prompt
        return f"{default_prompt}\n{stripped}"

    def _result_text(self, result: object) -> str:
        if isinstance(result, dict):
            text = result.get("text", "")
            return text if isinstance(text, str) else str(text)
        text = getattr(result, "text", "")
        return text if isinstance(text, str) else str(text)

    def _parse_segments(self, text: str, helpers: _MossHelpers) -> list[TranscriptionSegment]:
        try:
            parsed = helpers.parse_transcript(text)
        except Exception as exc:
            raise AsrError(
                503,
                "inference_failed",
                "MOSS transcript parser failed",
                {"error_type": type(exc).__name__, "message": str(exc)[-500:]},
            ) from exc
        return [self._segment_from_parsed(segment) for segment in parsed]

    def _segment_from_parsed(self, segment: object) -> TranscriptionSegment:
        if isinstance(segment, dict):
            return TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                speaker=str(segment["speaker"]) if segment.get("speaker") is not None else None,
                text=str(segment["text"]),
            )
        return TranscriptionSegment(
            start=float(getattr(segment, "start")),
            end=float(getattr(segment, "end")),
            speaker=str(getattr(segment, "speaker")) if getattr(segment, "speaker", None) is not None else None,
            text=str(getattr(segment, "text")),
        )

    def _close_model(self, model: Any) -> None:
        for method_name in ("close", "shutdown", "destroy", "cleanup"):
            cleanup = getattr(model, method_name, None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning("MOSS model cleanup method %s failed during unload: %s", method_name, exc)

    def _move_model_to_cpu(self, model: Any) -> None:
        to_device = getattr(model, "to", None)
        if callable(to_device):
            try:
                to_device("cpu")
            except Exception as exc:
                logger.warning("moving MOSS model to CPU failed during unload: %s", exc)

    def _release_cuda_cache(self) -> None:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return
        cuda = torch.cuda
        if not self._cuda_is_available(cuda):
            return
        self._run_cuda_cleanup("synchronize", cuda.synchronize)
        self._run_cuda_cleanup("empty_cache", cuda.empty_cache)
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            self._run_cuda_cleanup("ipc_collect", ipc_collect)

    def _cuda_is_available(self, cuda: Any) -> bool:
        try:
            return bool(cuda.is_available())
        except Exception as exc:
            logger.warning("torch.cuda.is_available failed during MOSS unload: %s", exc)
            return False

    def _run_cuda_cleanup(self, name: str, cleanup: Callable[[], object]) -> None:
        try:
            cleanup()
        except Exception as exc:
            logger.warning("torch.cuda.%s failed during MOSS unload: %s", name, exc)

    def _map_moss_exception(self, exc: Exception, *, phase: str, model_id: str) -> AsrError:
        message = str(exc)
        lowered = message.lower()
        details = {
            "phase": phase,
            "model": model_id,
            "error_type": type(exc).__name__,
            "message": message[-500:],
        }
        if "out of memory" in lowered or "cuda oom" in lowered:
            return AsrError(503, "gpu_unavailable", "CUDA out of memory during MOSS ASR", details)
        if phase == "load" and any(
            marker in lowered
            for marker in ("download", "connection", "repository", "resolve", "not found", "401", "403", "404")
        ):
            return AsrError(503, "model_download_failed", "MOSS model download or resolution failed", details)
        return AsrError(503, "inference_failed", f"MOSS {phase} failed", details)
