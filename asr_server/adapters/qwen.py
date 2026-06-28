from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

from asr_server.adapters.base import TranscriptionResult
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

    async def load(self, backend: str, device: str, dtype: str) -> None:
        del dtype
        if device != "cuda":
            raise AsrError(422, "capability_not_supported", "Qwen adapter requires device=cuda")
        self._assert_cuda_torch()
        qwen_asr = self._import_qwen_asr()
        repo_id = MODEL_REPOS[self.model_id]
        if backend == "transformers":
            self._model = qwen_asr.Qwen3ASRModel.from_pretrained(
                repo_id,
                device_map="cuda:0",
                max_inference_batch_size=1,
            )
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
    ) -> TranscriptionResult:
        del model_id, backend
        if self._model is None:
            raise AsrError(409, "model_loading", "Qwen model is not loaded")
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as audio_file:
            audio_file.write(audio)
            audio_path = Path(audio_file.name)
        try:
            qwen_language = None if language == "auto" else language
            results = self._model.transcribe(audio=str(audio_path), language=qwen_language)
            first = results[0]
            text = self._result_text(first)
            detected_language = self._result_language(first) or ("zh" if language == "auto" else language)
            return TranscriptionResult(
                text=text,
                duration=max(len(audio) / 16_000, 0.01),
                language=detected_language,
                warnings=[],
            )
        finally:
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

    def _result_language(self, result: Any) -> str:
        language = getattr(result, "language", "")
        if isinstance(language, str):
            return language
        return str(language)
