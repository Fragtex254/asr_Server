from __future__ import annotations

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
