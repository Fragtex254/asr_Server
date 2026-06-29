from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings
from asr_server.errors import AsrError


MODEL_REPOS = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}


class QwenAsrAdapter:
    """Lazy Qwen3-ASR adapter skeleton for WSL GPU deployment."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.loaded_backend: str | None = None
        self._model: Any | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del dtype
        if device != "cuda":
            raise AsrError(422, "capability_not_supported", "Qwen adapter requires device=cuda")
        self._assert_cuda_torch()
        qwen_asr = self._import_qwen_asr()
        repo_id = MODEL_REPOS[self.model_id]
        try:
            if backend == "transformers":
                load_kwargs: dict[str, Any] = {
                    "device_map": "cuda:0",
                    "max_inference_batch_size": 1,
                }
                if max_new_tokens is not None:
                    load_kwargs["max_new_tokens"] = max_new_tokens
                self._model = qwen_asr.Qwen3ASRModel.from_pretrained(repo_id, **load_kwargs)
            elif backend == "vllm":
                self._configure_vllm_environment()
                llm_kwargs: dict[str, Any] = {
                    "model": repo_id,
                    "gpu_memory_utilization": float(os.getenv("ASR_QWEN_VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
                    "max_inference_batch_size": 1,
                }
                max_model_len = os.getenv("ASR_QWEN_VLLM_MAX_MODEL_LEN", "32768")
                llm_kwargs["max_model_len"] = int(max_model_len)
                self._model = qwen_asr.Qwen3ASRModel.LLM(**llm_kwargs)
            else:
                raise AsrError(422, "capability_not_supported", f"unsupported Qwen backend: {backend}")
        except AsrError:
            raise
        except Exception as exc:
            raise self._map_qwen_exception(exc, phase="load", model_id=repo_id) from exc
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        self._model = None
        self.loaded_backend = None
        if cuda_empty_cache:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        if self._model is None:
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
            try:
                results = self._model.transcribe(**transcribe_kwargs)
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
                warnings=[],
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
        if self._model is None:
            raise AsrError(409, "model_loading", "Qwen model is not loaded")
        audio_paths: list[Path] = []
        try:
            for audio in audio_chunks:
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as audio_file:
                    audio_file.write(audio)
                    audio_paths.append(Path(audio_file.name))
            qwen_language = None if language == "auto" else language
            transcribe_kwargs: dict[str, Any] = {
                "audio": [str(audio_path) for audio_path in audio_paths],
                "language": [qwen_language] * len(audio_paths),
            }
            if context:
                transcribe_kwargs["context"] = [context] * len(audio_paths)
            inference_started = perf_counter()
            try:
                results = self._model.transcribe(**transcribe_kwargs)
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
                    warnings=[],
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

    def _import_qwen_asr(self) -> Any:
        try:
            return importlib.import_module("qwen_asr")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "gpu_unavailable",
                "qwen_asr is not installed; install qwen-asr in WSL after CUDA torch validation",
            ) from exc

    def _assert_cuda_torch(self) -> None:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise AsrError(503, "gpu_unavailable", "torch is not installed") from exc
        if torch.version.cuda is None:
            raise AsrError(503, "gpu_unavailable", "installed torch is CPU-only")
        if not torch.cuda.is_available():
            raise AsrError(503, "gpu_unavailable", "torch cannot access CUDA")

    def _configure_vllm_environment(self) -> None:
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

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
