from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, TypeVar

from asr_server.adapters.base import AsrAdapter, TranscriptionResult, TranscriptionTimings
from asr_server.errors import AsrError
from asr_server.registry import Backend, ModelDefinition, ModelStatus


ChunkProgressCallback = Callable[[int, int], Awaitable[None]]
ChunkResultCallback = Callable[[int, int, TranscriptionResult], Awaitable[None]]
T = TypeVar("T")


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
    max_new_tokens: int | None = None
    loaded_at: str | None = None
    last_used_at: str | None = None
    vram_allocated_mb: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idle_unload_task: asyncio.Task[None] | None = None

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
            "max_new_tokens": self.max_new_tokens,
            "loaded_at": self.loaded_at,
            "last_used_at": self.last_used_at,
            "vram_allocated_mb": self.vram_allocated_mb,
        }


class ModelLifecycleManager:
    def __init__(
        self,
        models: dict[str, ModelDefinition],
        adapter_factory: Callable[[str], AsrAdapter],
        idle_unload_seconds: float = 180.0,
    ) -> None:
        self._runtimes = {
            model_id: ModelRuntime(definition=model, adapter=adapter_factory(model_id))
            for model_id, model in models.items()
        }
        self._default_model_id = next(model.id for model in models.values() if model.default)
        self._idle_unload_seconds = idle_unload_seconds

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

    async def shutdown(self) -> None:
        for runtime in self._runtimes.values():
            self._cancel_idle_unload(runtime)

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
        max_new_tokens: int | None = None,
    ) -> dict[str, object]:
        runtime = self.runtime_for(model_id)
        resolved_backend = self.resolve_backend(runtime, backend)
        async with runtime.request_lock:
            async with runtime.lock:
                self._cancel_idle_unload(runtime)
                if runtime.status in ("unloading", "unloading_scheduled"):
                    raise AsrError(409, "model_unloading_scheduled", f"model is unloading: {model_id}")
                runtime.status = "loading"
                try:
                    await _call_adapter(
                        lambda: runtime.adapter.load(resolved_backend, device, dtype, max_new_tokens=max_new_tokens)
                    )
                except Exception:
                    runtime.status = "error"
                    runtime.rejecting_new_requests = False
                    raise
                runtime.status = "loaded"
                runtime.rejecting_new_requests = False
                runtime.backend = resolved_backend
                runtime.max_new_tokens = max_new_tokens
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
            self._cancel_idle_unload(runtime)
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
        context: str = "",
        max_new_tokens: int | None = None,
        batch_size: int = 1,
    ) -> tuple[str, TranscriptionResult]:
        resolved_backend, results, timings = await self.transcribe_chunks(
            [audio],
            model_id=model_id,
            backend=backend,
            language=language,
            timestamps=timestamps,
            context=context,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
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
        context: str,
        max_new_tokens: int | None,
        batch_size: int,
        before_chunk: ChunkProgressCallback | None = None,
        after_chunk: ChunkResultCallback | None = None,
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
        async with runtime.lock:
            if runtime.status == "unloading_scheduled" or runtime.rejecting_new_requests:
                raise AsrError(
                    409,
                    "model_unloading_scheduled",
                    f"model is scheduled for unloading: {runtime.definition.id}",
                )
        async with runtime.request_lock:
            resolved_backend, load_ms = await self._begin_request(
                runtime,
                backend=backend,
                max_new_tokens=max_new_tokens,
            )
            try:
                results = await self._transcribe_chunk_results(
                    runtime,
                    audio_chunks,
                    model_id=selected_model_id,
                    backend=resolved_backend,
                    language=language,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    before_chunk=before_chunk,
                    after_chunk=after_chunk,
                )
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

    async def _transcribe_chunk_results(
        self,
        runtime: ModelRuntime,
        audio_chunks: list[bytes],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
        batch_size: int,
        before_chunk: ChunkProgressCallback | None,
        after_chunk: ChunkResultCallback | None,
    ) -> list[TranscriptionResult]:
        if len(audio_chunks) == 1:
            if before_chunk is not None:
                await before_chunk(0, 1)
            chunk_started = perf_counter()
            result = await _call_adapter(
                lambda: runtime.adapter.transcribe(
                    audio_chunks[0],
                    model_id=model_id,
                    backend=backend,
                    language=language,
                    context=context,
                    max_new_tokens=max_new_tokens,
                )
            )
            if result.timings.total_ms == 0:
                result = replace(
                    result,
                    timings=replace(result.timings, total_ms=(perf_counter() - chunk_started) * 1000),
                )
            if after_chunk is not None:
                await after_chunk(0, 1, result)
            return [result]

        results = []
        effective_batch_size = max(batch_size, 1)
        completed_count = 0
        total_count = len(audio_chunks)
        for batch in _chunked(audio_chunks, effective_batch_size):
            batch_start_index = completed_count
            if before_chunk is not None:
                for offset in range(len(batch)):
                    await before_chunk(batch_start_index + offset, total_count)
            batch_started = perf_counter()
            try:
                batch_results = await _call_adapter(
                    lambda: runtime.adapter.transcribe_batch(
                        batch,
                        model_id=model_id,
                        backend=backend,
                        language=language,
                        context=context,
                        max_new_tokens=max_new_tokens,
                    )
                )
            except AsrError:
                raise
            except Exception as exc:
                raise AsrError(
                    503,
                    "inference_failed",
                    "batch transcription failed",
                    {"error_type": type(exc).__name__},
                ) from exc
            if len(batch_results) != len(batch):
                raise AsrError(
                    503,
                    "inference_failed",
                    "batch transcription returned an unexpected result count",
                    {"expected": len(batch), "actual": len(batch_results)},
                )
            batch_ms = (perf_counter() - batch_started) * 1000
            per_chunk_ms = batch_ms / len(batch)
            for offset, result in enumerate(batch_results):
                if result.timings.total_ms == 0:
                    result = replace(result, timings=replace(result.timings, total_ms=per_chunk_ms))
                results.append(result)
                completed_count += 1
                if after_chunk is not None:
                    await after_chunk(batch_start_index + offset, total_count, result)
        return results

    async def _begin_request(
        self,
        runtime: ModelRuntime,
        *,
        backend: Backend,
        max_new_tokens: int | None,
    ) -> tuple[str, float]:
        resolved_backend = self.resolve_backend(runtime, backend)
        load_ms = 0.0
        async with runtime.lock:
            self._cancel_idle_unload(runtime)
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
            needs_reload_for_generation = (
                runtime.status == "loaded"
                and runtime.backend == resolved_backend
                and runtime.max_new_tokens != max_new_tokens
            )
            if needs_reload_for_generation:
                await _call_adapter(lambda: runtime.adapter.unload(cuda_empty_cache=True))
                runtime.status = "unloaded"
                runtime.backend = None
                runtime.max_new_tokens = None
                runtime.loaded_at = None
            if runtime.status in ("unloaded", "error") or runtime.backend != resolved_backend:
                runtime.status = "loading"
                load_started = perf_counter()
                try:
                    await _call_adapter(
                        lambda: runtime.adapter.load(resolved_backend, "cuda", "auto", max_new_tokens=max_new_tokens)
                    )
                except Exception:
                    runtime.status = "error"
                    runtime.rejecting_new_requests = False
                    raise
                load_ms = (perf_counter() - load_started) * 1000
                runtime.status = "loaded"
                runtime.backend = resolved_backend
                runtime.max_new_tokens = max_new_tokens
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
            elif runtime.status == "loaded" and runtime.active_requests == 0:
                self._schedule_idle_unload(runtime)

    async def _unload_now(
        self,
        runtime: ModelRuntime,
        *,
        cuda_empty_cache: bool,
        cancel_idle_task: bool = True,
    ) -> None:
        if cancel_idle_task:
            self._cancel_idle_unload(runtime)
        runtime.status = "unloading"
        await _call_adapter(lambda: runtime.adapter.unload(cuda_empty_cache))
        runtime.status = "unloaded"
        runtime.rejecting_new_requests = False
        runtime.backend = None
        runtime.max_new_tokens = None
        runtime.loaded_at = None
        runtime.vram_allocated_mb = None

    def _schedule_idle_unload(self, runtime: ModelRuntime) -> None:
        if self._idle_unload_seconds <= 0:
            return
        self._cancel_idle_unload(runtime)
        runtime.idle_unload_task = asyncio.create_task(self._idle_unload_after_delay(runtime))
        runtime.idle_unload_task.add_done_callback(_consume_task_exception)

    def _cancel_idle_unload(self, runtime: ModelRuntime) -> None:
        task = runtime.idle_unload_task
        if task is None:
            return
        if task is not asyncio.current_task() and not task.done():
            task.cancel()
        runtime.idle_unload_task = None

    async def _idle_unload_after_delay(self, runtime: ModelRuntime) -> None:
        try:
            await asyncio.sleep(self._idle_unload_seconds)
            async with runtime.lock:
                if runtime.status != "loaded" or runtime.active_requests > 0 or runtime.rejecting_new_requests:
                    return
                try:
                    await self._unload_now(runtime, cuda_empty_cache=True, cancel_idle_task=False)
                except Exception:
                    runtime.status = "error"
                    runtime.rejecting_new_requests = False
                    raise
                finally:
                    runtime.idle_unload_task = None
        except asyncio.CancelledError:
            return


def _chunked(items: list[bytes], batch_size: int) -> list[list[bytes]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


async def _call_adapter(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    return await asyncio.to_thread(lambda: asyncio.run(factory()))


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
