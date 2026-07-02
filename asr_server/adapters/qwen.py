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

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings
from asr_server.errors import AsrError


MODEL_REPOS = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}
logger = logging.getLogger(__name__)


def _qwen_worker_main(conn: Connection, model_id: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = _QwenWorkerBackend(model_id)
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
                raise AsrError(400, "bad_request", f"unknown Qwen worker operation: {op}")
            except AsrError as exc:
                conn.send({"id": request_id, "ok": False, "error": _error_to_payload(exc)})
            except Exception as exc:
                error = AsrError(
                    503,
                    "inference_failed",
                    "Qwen worker operation failed",
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


class QwenAsrAdapter:
    """Qwen3-ASR adapter with GPU work isolated in a child process."""

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
                logger.warning("Qwen worker graceful shutdown failed: %s", exc.message)
            except (BrokenPipeError, EOFError, OSError) as exc:
                logger.warning("Qwen worker pipe closed during shutdown: %s", exc)
        if worker.is_alive():
            worker.join(timeout=10)
        if worker.is_alive():
            logger.warning("Qwen worker did not exit after unload; terminating pid=%s", worker.pid)
            worker.terminate()
            worker.join(timeout=5)
        if worker.is_alive():
            logger.warning("Qwen worker did not terminate cleanly; killing pid=%s", worker.pid)
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
        del model_id, backend
        payload = self._request(
            "transcribe",
            audio=audio,
            language=language,
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
        del model_id, backend
        payload = self._request(
            "transcribe_batch",
            audio_chunks=audio_chunks,
            language=language,
            context=context,
            max_new_tokens=max_new_tokens,
        )
        return [_result_from_payload(item) for item in payload]

    def _start_worker(self) -> None:
        parent_conn, child_conn = multiprocessing.get_context("spawn").Pipe()
        worker = multiprocessing.get_context("spawn").Process(
            target=_qwen_worker_main,
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
            raise AsrError(409, "model_loading", "Qwen worker is not running")
        return self._request_on(conn, op, **payload)

    def _request_on(self, conn: Connection, op: str, **payload: object) -> Any:
        self._request_id += 1
        request_id = self._request_id
        conn.send({"id": request_id, "op": op, **payload})
        response = conn.recv()
        if response.get("id") != request_id:
            raise AsrError(503, "inference_failed", "Qwen worker returned an unexpected response")
        if response.get("ok"):
            return response.get("result")
        error = response.get("error", {})
        raise AsrError(
            int(error.get("status_code", 503)),
            str(error.get("code", "inference_failed")),
            str(error.get("message", "Qwen worker failed")),
            dict(error.get("details", {})),
        )


class _QwenWorkerBackend:
    """In-process Qwen3-ASR backend used only inside the worker process."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.loaded_backend: str | None = None
        self._model: Any | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        if device not in {"cuda", "cuda:0"}:
            raise AsrError(422, "capability_not_supported", "Qwen HF native adapter requires device=cuda or cuda:0")
        if backend != "transformers":
            raise AsrError(422, "capability_not_supported", f"unsupported Qwen backend: {backend}")
        torch = self._assert_cuda_torch()
        torch_dtype = self._resolve_torch_dtype(torch, dtype)
        processor_cls, model_cls = self._import_hf_native_transformers()
        repo_id = MODEL_REPOS[self.model_id]
        if self._model is not None:
            await self.unload(cuda_empty_cache=True)
        try:
            processor = processor_cls.from_pretrained(repo_id)
            model = model_cls.from_pretrained(repo_id, dtype=torch_dtype).to(device).eval()
            self._model = _HfNativeQwenModel(
                processor=processor,
                model=model,
                torch=torch,
                max_new_tokens=max_new_tokens,
            )
        except AsrError:
            raise
        except Exception as exc:
            raise self._map_qwen_exception(exc, phase="load", model_id=repo_id) from exc
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        model = self._model
        self._model = None
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
        del model_id, backend
        model = self._model
        if model is None:
            raise AsrError(409, "model_loading", "Qwen model is not loaded")
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as audio_file:
            audio_file.write(audio)
            audio_path = Path(audio_file.name)
        try:
            qwen_language = None if language == "auto" else language
            inference_started = perf_counter()
            transcribe_kwargs: dict[str, Any] = {"audio": str(audio_path), "language": qwen_language}
            if context:
                transcribe_kwargs["context"] = context
            if max_new_tokens is not None:
                transcribe_kwargs["max_new_tokens"] = max_new_tokens
            try:
                results = self._transcribe_model(model, transcribe_kwargs)
            except Exception as exc:
                raise self._map_qwen_exception(exc, phase="inference", model_id=self.model_id) from exc
            inference_ms = (perf_counter() - inference_started) * 1000
            postprocess_started = perf_counter()
            first = self._first_result(results)
            text = self._result_text(first)
            if not text.strip():
                raise AsrError(503, "inference_failed", "Qwen returned an empty transcription result")
            detected_language = self._result_language(first) or ("zh" if language == "auto" else language)
            postprocess_ms = (perf_counter() - postprocess_started) * 1000
            return TranscriptionResult(
                text=text,
                duration=max(len(audio) / 16_000, 0.01),
                language=detected_language,
                warnings=self._result_warnings(first),
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
        del model_id, backend
        model = self._model
        if model is None:
            raise AsrError(409, "model_loading", "Qwen model is not loaded")
        audio_paths: list[Path] = []
        try:
            for audio in audio_chunks:
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as audio_file:
                    audio_file.write(audio)
                    audio_paths.append(Path(audio_file.name))
            qwen_language = None if language == "auto" else language
            inference_started = perf_counter()
            try:
                results = [
                    self._transcribe_model(
                        model,
                        {
                            "audio": str(audio_path),
                            "language": qwen_language,
                            "context": context,
                            "max_new_tokens": max_new_tokens,
                        },
                    )[0]
                    for audio_path in audio_paths
                ]
            except Exception as exc:
                raise self._map_qwen_exception(exc, phase="inference", model_id=self.model_id) from exc
            inference_ms = (perf_counter() - inference_started) * 1000
            if len(results) != len(audio_chunks):
                raise AsrError(
                    503,
                    "inference_failed",
                    "Qwen batch transcription returned an unexpected result count",
                    {"expected": len(audio_chunks), "actual": len(results)},
                )
            postprocess_started = perf_counter()
            transcriptions = [
                TranscriptionResult(
                    text=self._non_empty_text(result),
                    duration=max(len(audio) / 16_000, 0.01),
                    language=self._result_language(result) or ("zh" if language == "auto" else language),
                    warnings=self._result_warnings(result),
                    timings=TranscriptionTimings(
                        inference_ms=inference_ms / len(audio_chunks) if audio_chunks else 0.0,
                        postprocess_ms=0.0,
                    ),
                )
                for audio, result in zip(audio_chunks, results, strict=True)
            ]
            postprocess_ms = (perf_counter() - postprocess_started) * 1000
            if transcriptions:
                per_chunk_postprocess_ms = postprocess_ms / len(transcriptions)
                transcriptions = [
                    TranscriptionResult(
                        text=result.text,
                        duration=result.duration,
                        language=result.language,
                        warnings=result.warnings,
                        timings=TranscriptionTimings(
                            inference_ms=result.timings.inference_ms,
                            postprocess_ms=per_chunk_postprocess_ms,
                        ),
                    )
                    for result in transcriptions
                ]
            return transcriptions
        finally:
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)

    def _import_hf_native_transformers(self) -> tuple[Any, Any]:
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "gpu_unavailable",
                "transformers is not installed; install HF native dependencies in WSL after CUDA torch validation",
            ) from exc
        processor_cls = getattr(transformers, "AutoProcessor", None)
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if processor_cls is None or model_cls is None:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "installed transformers does not provide Qwen3-ASR HF native classes; install a newer release or transformers main",
                {
                    "missing": [
                        name
                        for name, value in (
                            ("AutoProcessor", processor_cls),
                            ("AutoModelForMultimodalLM", model_cls),
                        )
                        if value is None
                    ]
                },
            )
        return processor_cls, model_cls

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
            "Qwen HF native adapter supports dtype=auto, bfloat16/bf16, or float16/fp16",
            {"dtype": dtype},
        )

    def _transcribe_model(self, model: Any, transcribe_kwargs: dict[str, Any]) -> Any:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return model.transcribe(**transcribe_kwargs)
        inference_mode = getattr(torch, "inference_mode", None)
        if not callable(inference_mode):
            return model.transcribe(**transcribe_kwargs)
        try:
            inference_context = inference_mode()
        except Exception as exc:
            logger.warning("torch.inference_mode setup failed during Qwen transcribe: %s", exc)
            return model.transcribe(**transcribe_kwargs)
        with inference_context:
            return model.transcribe(**transcribe_kwargs)

    def _close_model(self, model: Any) -> None:
        for method_name in ("close", "shutdown", "destroy", "cleanup"):
            cleanup = getattr(model, method_name, None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning("Qwen model cleanup method %s failed during unload: %s", method_name, exc)

    def _move_model_to_cpu(self, model: Any) -> None:
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(model, "forced_aligner", None),
            getattr(getattr(model, "forced_aligner", None), "model", None),
        ]
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            to_device = getattr(candidate, "to", None)
            if callable(to_device):
                try:
                    to_device("cpu")
                except Exception as exc:
                    logger.warning("moving Qwen model component to CPU failed during unload: %s", exc)

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
            logger.warning("torch.cuda.is_available failed during unload: %s", exc)
            return False

    def _run_cuda_cleanup(self, name: str, cleanup: Callable[[], object]) -> None:
        try:
            cleanup()
        except Exception as exc:
            logger.warning("torch.cuda.%s failed during unload: %s", name, exc)

    def _result_text(self, result: Any) -> str:
        text = getattr(result, "text", "")
        if isinstance(text, str):
            return text
        return str(text)

    def _non_empty_text(self, result: Any) -> str:
        text = self._result_text(result)
        if not text.strip():
            raise AsrError(503, "inference_failed", "Qwen returned an empty transcription result")
        return text

    def _result_language(self, result: Any) -> str:
        language = getattr(result, "language", "")
        if isinstance(language, str):
            return language
        return str(language)

    def _result_warnings(self, result: Any) -> list[str]:
        warnings = getattr(result, "warnings", [])
        if not isinstance(warnings, list):
            return [str(warnings)]
        return [str(warning) for warning in warnings]

    def _first_result(self, results: Any) -> Any:
        try:
            first = results[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise AsrError(503, "inference_failed", "Qwen returned no transcription results") from exc
        return first

    def _map_qwen_exception(self, exc: Exception, *, phase: str, model_id: str) -> AsrError:
        message = str(exc)
        lowered = message.lower()
        details = {
            "phase": phase,
            "model": model_id,
            "error_type": type(exc).__name__,
            "message": message[-500:],
        }
        if "out of memory" in lowered or "cuda oom" in lowered:
            return AsrError(503, "gpu_unavailable", "CUDA out of memory during Qwen ASR", details)
        if phase == "load" and any(
            marker in lowered
            for marker in ("download", "connection", "repository", "resolve", "not found", "401", "403", "404")
        ):
            return AsrError(503, "model_download_failed", "Qwen model download or resolution failed", details)
        return AsrError(503, "inference_failed", f"Qwen {phase} failed", details)


@dataclass(frozen=True)
class _QwenTranscriptionItem:
    text: str
    language: str
    warnings: list[str]


class _HfNativeQwenModel:
    """Small adapter around the Qwen3-ASR Transformers native helper API."""

    def __init__(self, *, processor: Any, model: Any, torch: Any, max_new_tokens: int | None) -> None:
        self.processor = processor
        self.model = model
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def transcribe(self, **kwargs: object) -> list[_QwenTranscriptionItem]:
        audio = kwargs.get("audio")
        if not isinstance(audio, str):
            raise TypeError("HF native Qwen transcription expects one audio path")
        language_value = kwargs.get("language")
        language = language_value if isinstance(language_value, str) else None
        request_max_new_tokens = kwargs.get("max_new_tokens")
        max_new_tokens = request_max_new_tokens if isinstance(request_max_new_tokens, int) else self.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = 512
        context = kwargs.get("context")
        warnings: list[str] = []
        if isinstance(context, str) and context:
            warnings.append("context is not applied by the HF native transcription helper")

        apply_request = getattr(self.processor, "apply_transcription_request", None)
        if not callable(apply_request):
            raise RuntimeError("processor.apply_transcription_request is unavailable")
        inputs = apply_request(audio=audio, language=language)
        inputs = inputs.to(self.model.device, self.model.dtype)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed_items = self.processor.decode(generated_ids, return_format="parsed")
        return [self._item_from_parsed(item, warnings=warnings) for item in parsed_items]

    def _item_from_parsed(self, parsed: object, *, warnings: list[str]) -> _QwenTranscriptionItem:
        if isinstance(parsed, dict):
            text = parsed.get("transcription", parsed.get("text", ""))
            language = parsed.get("language", "")
            return _QwenTranscriptionItem(
                text=text if isinstance(text, str) else str(text),
                language=language if isinstance(language, str) else str(language),
                warnings=warnings,
            )
        return _QwenTranscriptionItem(text=str(parsed), language="", warnings=warnings)
