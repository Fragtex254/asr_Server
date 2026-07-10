from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from asr_server.adapters.base import TranscriptionResult
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.audio.workspace import AudioWorkspace
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import Backend
from asr_server.transcription import (
    TranscriptionRequest,
    ValidatedTranscription,
    run_transcription_path,
    validate_transcription_request,
)


JobStatus = Literal[
    "queued",
    "preprocessing",
    "splitting",
    "loading_model",
    "transcribing",
    "merging",
    "completed",
    "failed",
    "cancel_requested",
    "cancelled",
    "expired",
]
logger = logging.getLogger(__name__)

TERMINAL_STATUSES: set[JobStatus] = {"completed", "failed", "cancelled", "expired"}


class JobCancelled(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


@dataclass
class JobProgress:
    phase: str
    percent: float = 0.0
    total_chunks: int | None = None
    completed_chunks: int | None = None
    current_chunk: int | None = None
    current_chunk_start: float | None = None
    current_chunk_end: float | None = None
    message: str | None = None

    def to_api(self) -> dict[str, object]:
        payload: dict[str, object] = {"phase": self.phase, "percent": self.percent}
        if self.total_chunks is not None:
            payload["total_chunks"] = self.total_chunks
        if self.completed_chunks is not None:
            payload["completed_chunks"] = self.completed_chunks
        if self.current_chunk is not None:
            payload["current_chunk"] = self.current_chunk
        if self.current_chunk_start is not None:
            payload["current_chunk_start"] = self.current_chunk_start
        if self.current_chunk_end is not None:
            payload["current_chunk_end"] = self.current_chunk_end
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass
class TranscriptionJob:
    id: str
    status: JobStatus
    model: str
    backend: str
    language: str
    request: TranscriptionRequest
    validated: ValidatedTranscription
    upload_path: Path
    work_dir: Path
    created_at: datetime
    expires_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: JobProgress = field(default_factory=lambda: JobProgress(phase="queued", message="waiting for previous transcription jobs"))
    split: dict[str, object] | None = None
    result: dict[str, object] | None = None
    error: dict[str, object] | None = None
    request_summary: dict[str, object] = field(default_factory=dict)
    temp_paths: list[Path] = field(default_factory=list)
    workspace: AudioWorkspace | None = None
    chunk_windows: list[tuple[float, float]] = field(default_factory=list)


class JobManager:
    def __init__(self, manager: ModelLifecycleManager, settings: Settings) -> None:
        self._manager = manager
        self._settings = settings
        self._jobs: dict[str, TranscriptionJob] = {}
        self._queue: deque[str] = deque()
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._closed = False
        self._stale_temp_dirs_cleaned = False

    async def start(self) -> None:
        if not self._stale_temp_dirs_cleaned:
            await asyncio.to_thread(self._cleanup_stale_temp_dirs)
            self._stale_temp_dirs_cleaned = True
        if self._worker is None or self._worker.done():
            self._closed = False
            self._worker = asyncio.create_task(self._worker_loop())
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._expiry_loop())

    async def shutdown(self) -> None:
        self._closed = True
        async with self._condition:
            queued_ids = list(self._queue)
            self._queue.clear()
            for job_id in queued_ids:
                job = self._jobs.get(job_id)
                if job is not None:
                    self._mark_cancelled(job)
            for job in self._jobs.values():
                if job.id not in queued_ids and job.status not in TERMINAL_STATUSES:
                    job.status = "cancel_requested"
            self._condition.notify_all()
        for job_id in queued_ids:
            job = self._jobs.get(job_id)
            if job is not None:
                await self._cleanup_job_files(job)
        if self._worker is not None:
            done, _pending = await asyncio.wait(
                {self._worker},
                timeout=self._settings.job_shutdown_grace_seconds,
            )
            if not done:
                await self._manager.abort_all_workers()
                self._worker.cancel()
                try:
                    await self._worker
                except asyncio.CancelledError:
                    pass
            self._worker = None
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None
        for job in list(self._jobs.values()):
            await self._cleanup_job_files(job)

    async def create_job(
        self,
        *,
        workspace: AudioWorkspace,
        filename: str | None,
        content_type: str | None,
        request: TranscriptionRequest,
    ) -> dict[str, object]:
        await self.start()
        self._expire_jobs()
        validated = validate_transcription_request(self._manager, request)
        job_id = f"job_{uuid4().hex}"
        created_at = utc_now()
        upload_name = filename or "upload.bin"
        async with self._condition:
            if self._active_job_count() >= self._settings.max_queued_jobs:
                raise AsrError(
                    429,
                    "job_queue_full",
                    "too many queued or running transcription jobs",
                    {"max_queued_jobs": self._settings.max_queued_jobs},
                )
            work_dir = workspace.root
            upload_path = workspace.upload_path
            job = TranscriptionJob(
                id=job_id,
                status="queued",
                model=validated.selected_model,
                backend=validated.resolved_backend,
                language=request.language,
                request=request,
                validated=validated,
                upload_path=upload_path,
                work_dir=work_dir,
                created_at=created_at,
                request_summary={
                    "filename": upload_name,
                    "content_type": content_type,
                    "size_bytes": workspace.size_bytes,
                    "response_format": request.response_format,
                    "timestamps": request.timestamps,
                    "split_strategy": request.split_strategy,
                    "max_chunk_seconds": request.max_chunk_seconds,
                    "overlap_seconds": request.overlap_seconds,
                    "preserve_segments": request.preserve_segments,
                    "context_chars": len(request.context),
                    "hotwords_chars": len(request.hotwords or ""),
                },
                temp_paths=[upload_path, work_dir],
                workspace=workspace,
            )
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._condition.notify()
            return self._job_create_payload(job)

    async def get_job(self, job_id: str) -> dict[str, object]:
        self._expire_jobs()
        job = self._job_or_404(job_id)
        return self._job_payload(job)

    async def cancel_job(self, job_id: str) -> dict[str, object]:
        self._expire_jobs()
        job = self._job_or_404(job_id)
        async with self._condition:
            if job.status == "queued":
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                self._mark_cancelled(job)
                await self._cleanup_job_files(job)
                self._condition.notify()
                return {"id": job.id, "status": job.status, "message": "job cancelled"}
            if job.status in TERMINAL_STATUSES:
                return {"id": job.id, "status": job.status, "message": f"job is already {job.status}"}
            job.status = "cancel_requested"
            job.progress.message = "cancellation will take effect after the current chunk finishes"
            return {
                "id": job.id,
                "status": job.status,
                "message": "cancellation will take effect after the current chunk finishes",
            }

    def _job_or_404(self, job_id: str) -> TranscriptionJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise AsrError(404, "job_not_found", f"unknown job: {job_id}") from exc

    async def _worker_loop(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or bool(self._queue))
                if self._closed and not self._queue:
                    return
                job_id = self._queue.popleft()
                job = self._jobs[job_id]
            await self._run_job(job)

    async def _run_job(self, job: TranscriptionJob) -> None:
        logger.info("starting transcription job id=%s model=%s", job.id, job.model)
        job.started_at = utc_now()
        job.status = "preprocessing"
        job.progress = JobProgress(phase="preprocessing", percent=1.0, message="preprocessing uploaded audio")
        try:
            async def stage_callback(phase: str, update: dict[str, object]) -> None:
                self._raise_if_cancel_requested(job)
                self._update_phase(job, phase, update)

            async def before_chunk(chunk_index: int, total_chunks: int) -> None:
                self._raise_if_cancel_requested(job)
                current = chunk_index + 1
                completed = max(current - 1, 0)
                percent = (completed / total_chunks) * 100 if total_chunks else 0.0
                job.status = "transcribing" if job.status != "cancel_requested" else "cancel_requested"
                job.progress = JobProgress(
                    phase="transcribing",
                    percent=percent,
                    total_chunks=total_chunks,
                    completed_chunks=completed,
                    current_chunk=current,
                    current_chunk_start=job.chunk_windows[chunk_index][0] if chunk_index < len(job.chunk_windows) else None,
                    current_chunk_end=job.chunk_windows[chunk_index][1] if chunk_index < len(job.chunk_windows) else None,
                    message=f"transcribing chunk {current} of {total_chunks}",
                )

            async def after_chunk(chunk_index: int, total_chunks: int, result: TranscriptionResult) -> None:
                del result
                completed = chunk_index + 1
                percent = (completed / total_chunks) * 100 if total_chunks else 100.0
                job.progress = JobProgress(
                    phase="transcribing",
                    percent=percent,
                    total_chunks=total_chunks,
                    completed_chunks=completed,
                    current_chunk=completed,
                    message=f"completed chunk {completed} of {total_chunks}",
                )
                self._raise_if_cancel_requested(job)

            result = await run_transcription_path(
                job.upload_path,
                workspace=job.work_dir,
                manager=self._manager,
                settings=self._settings,
                request=job.request,
                validated=job.validated,
                stage_callback=stage_callback,
                before_chunk=before_chunk,
                after_chunk=after_chunk,
            )
        except JobCancelled:
            self._mark_cancelled(job)
        except AsrError as exc:
            job.status = "failed"
            job.completed_at = utc_now()
            job.expires_at = job.completed_at + timedelta(seconds=self._settings.job_result_ttl_seconds)
            job.progress.phase = "failed"
            job.error = {"code": exc.code, "message": exc.message, "details": exc.details}
        except Exception as exc:
            job.status = "failed"
            job.completed_at = utc_now()
            job.expires_at = job.completed_at + timedelta(seconds=self._settings.job_result_ttl_seconds)
            job.progress.phase = "failed"
            job.error = {
                "code": "inference_failed",
                "message": "transcription job failed",
                "details": {"error_type": type(exc).__name__},
            }
        else:
            job.status = "completed"
            job.result = result
            split = result.get("split")
            job.split = split if isinstance(split, dict) else None
            job.completed_at = utc_now()
            job.expires_at = job.completed_at + timedelta(seconds=self._settings.job_result_ttl_seconds)
            split_count = _int_from_progress(job.progress.total_chunks)
            job.progress = JobProgress(
                phase="completed",
                percent=100.0,
                total_chunks=split_count,
                completed_chunks=split_count,
            )
        finally:
            await self._cleanup_job_files(job)
            logger.info("finished transcription job id=%s status=%s", job.id, job.status)

    def _raise_if_cancel_requested(self, job: TranscriptionJob) -> None:
        if job.status == "cancel_requested":
            raise JobCancelled()

    def _update_phase(self, job: TranscriptionJob, phase: str, update: dict[str, object]) -> None:
        if phase in {"preprocessing", "splitting", "loading_model", "merging"}:
            job.status = phase  # type: ignore[assignment]
        split = update.get("split")
        if isinstance(split, dict):
            job.split = split
        chunk_windows = update.get("chunk_windows")
        if isinstance(chunk_windows, list):
            parsed_windows: list[tuple[float, float]] = []
            for item in chunk_windows:
                if isinstance(item, tuple) and len(item) == 2:
                    parsed_windows.append((float(item[0]), float(item[1])))
            job.chunk_windows = parsed_windows
        percent = _optional_float(update.get("percent")) or job.progress.percent
        total_chunks = _optional_int(update.get("total_chunks"))
        completed_chunks = _optional_int(update.get("completed_chunks"))
        job.progress = JobProgress(
            phase=phase,
            percent=percent,
            total_chunks=total_chunks if total_chunks is not None else job.progress.total_chunks,
            completed_chunks=completed_chunks if completed_chunks is not None else job.progress.completed_chunks,
            message=_phase_message(phase),
        )

    def _mark_cancelled(self, job: TranscriptionJob) -> None:
        job.status = "cancelled"
        job.completed_at = utc_now()
        job.expires_at = job.completed_at + timedelta(seconds=self._settings.job_result_ttl_seconds)
        job.progress.phase = "cancelled"
        job.progress.message = "job cancelled"

    def _job_create_payload(self, job: TranscriptionJob) -> dict[str, object]:
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "backend": job.backend,
            "queue_position": self._queue_position(job.id),
            "created_at": utc_iso(job.created_at),
            "status_url": f"/v1/jobs/{job.id}",
        }

    def _job_payload(self, job: TranscriptionJob) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "backend": job.backend,
            "queue_position": self._queue_position(job.id),
            "progress": job.progress.to_api(),
            "created_at": utc_iso(job.created_at),
            "started_at": utc_iso(job.started_at),
            "completed_at": utc_iso(job.completed_at),
            "expires_at": utc_iso(job.expires_at),
            "elapsed_seconds": self._elapsed_seconds(job),
            "request": job.request_summary,
        }
        if job.split is not None:
            payload["split"] = job.split
        if job.status in {"completed", "failed", "cancelled", "expired"}:
            payload["result"] = job.result
            payload["error"] = job.error
        return payload

    def _queue_position(self, job_id: str) -> int:
        if job_id in self._queue:
            return list(self._queue).index(job_id) + 1
        return 0

    def _active_job_count(self) -> int:
        return sum(1 for job in self._jobs.values() if job.status not in TERMINAL_STATUSES)

    def _elapsed_seconds(self, job: TranscriptionJob) -> float:
        end = job.completed_at or utc_now()
        return max((end - job.created_at).total_seconds(), 0.0)

    def _expire_jobs(self) -> None:
        now = utc_now()
        for job_id, job in list(self._jobs.items()):
            if job.expires_at is None:
                continue
            if job.expires_at <= now:
                job.result = None
                job.error = None
                del self._jobs[job_id]

    async def _cleanup_job_files(self, job: TranscriptionJob) -> None:
        if job.workspace is not None:
            await job.workspace.cleanup()
            return
        for path in sorted(job.temp_paths, key=lambda item: len(item.parts), reverse=True):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup_stale_temp_dirs(self) -> None:
        temp_dir = Path(tempfile.gettempdir())
        for path in temp_dir.glob("asr_job_*"):
            try:
                if path.is_dir():
                    owner_path = path / "owner.pid"
                    if owner_path.exists():
                        try:
                            owner_pid = int(owner_path.read_text(encoding="ascii").strip())
                        except (OSError, ValueError):
                            owner_pid = -1
                        if owner_pid > 0 and Path(f"/proc/{owner_pid}").exists():
                            continue
                    shutil.rmtree(path)
            except OSError:
                pass

    async def _expiry_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(min(max(self._settings.job_result_ttl_seconds / 2, 1), 60))
                self._expire_jobs()
        except asyncio.CancelledError:
            return


def _phase_message(phase: str) -> str:
    messages = {
        "queued": "waiting for previous transcription jobs",
        "preprocessing": "preprocessing uploaded audio",
        "splitting": "splitting audio into chunks",
        "loading_model": "loading or confirming model",
        "merging": "merging chunk transcripts",
    }
    return messages.get(phase, phase)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_from_progress(value: int | None) -> int | None:
    return value
