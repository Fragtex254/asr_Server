from __future__ import annotations

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from asr_server.errors import AsrError


@dataclass(frozen=True)
class NormalizedAudio:
    audio: bytes
    decode_ms: float


@dataclass(frozen=True)
class NormalizedAudioPath:
    path: Path
    decode_ms: float


def probe_audio_duration_seconds(audio: bytes) -> float | None:
    input_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as input_file:
            input_file.write(audio)
            input_path = Path(input_file.name)
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        return None
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def probe_audio_duration_path(path: Path, *, timeout_seconds: float = 30.0) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired as exc:
        raise AsrError(408, "audio_probe_timeout", "ffprobe exceeded its operation deadline") from exc
    if completed.returncode != 0:
        return None
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


async def probe_audio_duration_path_async(path: Path, *, timeout_seconds: float = 30.0) -> float | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None
    try:
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        await _stop_subprocess(process)
        raise AsrError(408, "audio_probe_timeout", "ffprobe exceeded its operation deadline") from exc
    except asyncio.CancelledError:
        await _stop_subprocess(process)
        raise
    if process.returncode != 0:
        return None
    try:
        duration = float(stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def normalize_audio_to_wav(audio: bytes) -> NormalizedAudio:
    started = perf_counter()
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as input_file:
            input_file.write(audio)
            input_path = Path(input_file.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = Path(output_file.name)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AsrError(503, "audio_preprocess_unavailable", "ffmpeg is not installed") from exc
    try:
        decode_ms = (perf_counter() - started) * 1000
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise AsrError(
                400,
                "audio_decode_failed",
                "audio decode failed",
                {"ffmpeg": message[-1000:]},
            )
        if output_path is None:
            raise AsrError(500, "internal_error", "audio preprocess output path was not created")
        return NormalizedAudio(audio=output_path.read_bytes(), decode_ms=decode_ms)
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def normalize_audio_path_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    timeout_seconds: float = 1800.0,
) -> NormalizedAudioPath:
    started = perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise AsrError(503, "audio_preprocess_unavailable", "ffmpeg is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        output_path.unlink(missing_ok=True)
        raise AsrError(408, "audio_preprocess_timeout", "ffmpeg exceeded its operation deadline") from exc
    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AsrError(400, "audio_decode_failed", "audio decode failed", {"ffmpeg": message[-1000:]})
    return NormalizedAudioPath(path=output_path, decode_ms=(perf_counter() - started) * 1000)


async def normalize_audio_path_to_wav_async(
    input_path: Path,
    output_path: Path,
    *,
    timeout_seconds: float = 1800.0,
) -> NormalizedAudioPath:
    started = perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AsrError(503, "audio_preprocess_unavailable", "ffmpeg is not installed") from exc
    try:
        _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        await _stop_subprocess(process)
        output_path.unlink(missing_ok=True)
        raise AsrError(408, "audio_preprocess_timeout", "ffmpeg exceeded its operation deadline") from exc
    except asyncio.CancelledError:
        await _stop_subprocess(process)
        output_path.unlink(missing_ok=True)
        raise
    if process.returncode != 0:
        output_path.unlink(missing_ok=True)
        message = stderr.decode("utf-8", errors="replace").strip()
        raise AsrError(400, "audio_decode_failed", "audio decode failed", {"ffmpeg": message[-1000:]})
    return NormalizedAudioPath(path=output_path, decode_ms=(perf_counter() - started) * 1000)


async def _stop_subprocess(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()
