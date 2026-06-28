from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter

from asr_server.adapters.base import AsrAdapter, TranscriptionResult, TranscriptionTimings
from asr_server.errors import AsrError
from asr_server.registry import Backend, ModelDefinition, ModelStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelRuntime:
    definition: ModelDefinition
    adapter: AsrAdapter
    status: ModelStatus = "unloaded"
    active_requests: int = 0
    rejecting_new_requests: bool = False
    backend: str | None = None
    loaded_at: str | None = None
    last_used_at: str | None = None
    vram_allocated_mb: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def model_summary(self) -> dict[str, object]:
        return {
            "id": self.definition.id,
            "provider": self.definition.provider,
            "status": self.status,
            "default": self.definition.default,
            "capabilities": self.definition.capabilities.to_api(),
        }

    def status_summary(self) -> dict[str, object]:
        return {
            "id": self.definition.id,
            "status": self.status,
            "active_requests": self.active_requests,
            "rejecting_new_requests": self.rejecting_new_requests,
            "backend": self.backend,
            "loaded_at": self.loaded_at,
            "last_used_at": self.last_used_at,
            "vram_allocated_mb": self.vram_allocated_mb,
        }


class ModelLifecycleManager:
    def __init__(
        self,
        models: dict[str, ModelDefinition],
        adapter_factory: Callable[[str], AsrAdapter],
    ) -> None:
        self._runtimes = {
            model_id: ModelRuntime(definition=model, adapter=adapter_factory(model_id))
            for model_id, model in models.items()
        }
        self._default_model_id = next(model.id for model in models.values() if model.default)

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    def list_models(self) -> list[dict[str, object]]:
        return [runtime.model_summary() for runtime in self._runtimes.values()]

    def runtime_for(self, model_id: str) -> ModelRuntime:
        try:
            return self._runtimes[model_id]
        except KeyError as exc:
            raise AsrError(404, "model_not_found", f"unknown model: {model_id}") from exc

    def resolve_backend(self, runtime: ModelRuntime, backend: Backend) -> str:
        if backend == "auto":
            if runtime.backend is not None:
                return runtime.backend
            return runtime.definition.capabilities.backends[0]
        if backend not in runtime.definition.capabilities.backends:
            raise AsrError(
                422,
                "capability_not_supported",
                f"{runtime.definition.id} does not support backend: {backend}",
                {"backend": backend},
            )
        return backend

    async def load_model(
        self,
        model_id: str,
        *,
        backend: Backend = "auto",
        device: str = "cuda",
        dtype: str = "auto",
    ) -> dict[str, object]:
        runtime = self.runtime_for(model_id)
        resolved_backend = self.resolve_backend(runtime, backend)
        async with runtime.lock:
            if runtime.status in ("unloading", "unloading_scheduled"):
                raise AsrError(409, "model_unloading_scheduled", f"model is unloading: {model_id}")
            runtime.status = "loading"
            await runtime.adapter.load(resolved_backend, device, dtype)
            runtime.status = "loaded"
            runtime.rejecting_new_requests = False
            runtime.backend = resolved_backend
            runtime.loaded_at = utc_now_iso()
            return {
                "id": model_id,
                "status": runtime.status,
                "message": "model loaded",
            }

    async def unload_model(
        self,
        model_id: str,
        *,
        cuda_empty_cache: bool = True,
    ) -> dict[str, object]:
        runtime = self.runtime_for(model_id)
        async with runtime.lock:
            if runtime.active_requests > 0:
                runtime.status = "unloading_scheduled"
                runtime.rejecting_new_requests = True
                return {
                    "id": model_id,
                    "status": runtime.status,
                    "active_requests": runtime.active_requests,
                    "rejecting_new_requests": runtime.rejecting_new_requests,
                }
            await self._unload_now(runtime, cuda_empty_cache=cuda_empty_cache)
            return {
                "id": model_id,
                "status": runtime.status,
                "active_requests": runtime.active_requests,
                "rejecting_new_requests": runtime.rejecting_new_requests,
            }

    async def unload_all(self, *, cuda_empty_cache: bool = True) -> dict[str, object]:
        models = []
        for model_id in self._runtimes:
            models.append(await self.unload_model(model_id, cuda_empty_cache=cuda_empty_cache))
        return {"status": "accepted", "models": models}

    async def transcribe(
        self,
        audio: bytes,
        *,
        model_id: str | None,
        backend: Backend,
        language: str,
        timestamps: str,
    ) -> tuple[str, TranscriptionResult]:
        resolved_backend, results, timings = await self.transcribe_chunks(
            [audio],
            model_id=model_id,
            backend=backend,
            language=language,
            timestamps=timestamps,
        )
        return resolved_backend, replace(results[0], timings=timings)

    async def transcribe_chunks(
        self,
        audio_chunks: list[bytes],
        *,
        model_id: str | None,
        backend: Backend,
        language: str,
        timestamps: str,
    ) -> tuple[str, list[TranscriptionResult], TranscriptionTimings]:
        if not audio_chunks:
            raise AsrError(400, "bad_request", "audio request did not contain any chunks")
        request_started = perf_counter()
        selected_model_id = model_id or self.default_model_id
        runtime = self.runtime_for(selected_model_id)
        if timestamps != "none" and not runtime.definition.capabilities.timestamps:
            raise AsrError(
                422,
                "capability_not_supported",
                f"{selected_model_id} does not support timestamps in this server",
                {"timestamps": timestamps},
            )
        resolved_backend, load_ms = await self._begin_request(runtime, backend=backend)
        try:
            results = []
            for audio in audio_chunks:
                chunk_started = perf_counter()
                result = await runtime.adapter.transcribe(
                    audio,
                    model_id=selected_model_id,
                    backend=resolved_backend,
                    language=language,
                )
                if result.timings.total_ms == 0:
                    result = replace(
                        result,
                        timings=replace(result.timings, total_ms=(perf_counter() - chunk_started) * 1000),
                    )
                results.append(result)
            runtime.last_used_at = utc_now_iso()
            timings = TranscriptionTimings(
                total_ms=(perf_counter() - request_started) * 1000,
                load_ms=load_ms,
                decode_ms=sum(result.timings.decode_ms for result in results),
                inference_ms=sum(result.timings.inference_ms for result in results),
                postprocess_ms=sum(result.timings.postprocess_ms for result in results),
            )
            return resolved_backend, results, timings
        finally:
            await self._finish_request(runtime)

    async def _begin_request(self, runtime: ModelRuntime, *, backend: Backend) -> tuple[str, float]:
        resolved_backend = self.resolve_backend(runtime, backend)
        load_ms = 0.0
        async with runtime.lock:
            if runtime.status == "unloading_scheduled" or runtime.rejecting_new_requests:
                raise AsrError(
                    409,
                    "model_unloading_scheduled",
                    f"model is scheduled for unloading: {runtime.definition.id}",
                )
            if runtime.status == "loading":
                raise AsrError(409, "model_loading", f"model is loading: {runtime.definition.id}")
            if runtime.status == "unloading":
                raise AsrError(
                    409,
                    "model_unloading_scheduled",
                    f"model is unloading: {runtime.definition.id}",
                )
            if runtime.status in ("unloaded", "error") or runtime.backend != resolved_backend:
                runtime.status = "loading"
                load_started = perf_counter()
                await runtime.adapter.load(resolved_backend, "cuda", "auto")
                load_ms = (perf_counter() - load_started) * 1000
                runtime.status = "loaded"
                runtime.backend = resolved_backend
                runtime.loaded_at = utc_now_iso()
                runtime.rejecting_new_requests = False
            runtime.active_requests += 1
            return resolved_backend, load_ms

    async def _finish_request(self, runtime: ModelRuntime) -> None:
        async with runtime.lock:
            runtime.active_requests = max(runtime.active_requests - 1, 0)
            should_unload = runtime.status == "unloading_scheduled" and runtime.active_requests == 0
            if should_unload:
                await self._unload_now(runtime, cuda_empty_cache=True)

    async def _unload_now(self, runtime: ModelRuntime, *, cuda_empty_cache: bool) -> None:
        runtime.status = "unloading"
        await runtime.adapter.unload(cuda_empty_cache)
        runtime.status = "unloaded"
        runtime.rejecting_new_requests = False
        runtime.backend = None
        runtime.loaded_at = None
        runtime.vram_allocated_mb = None
