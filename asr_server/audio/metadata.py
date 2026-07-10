from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path


RAW_FALLBACK_BYTES_PER_SECOND = 16_000


@dataclass(frozen=True)
class AudioMetadata:
    duration_seconds: float
    format: str
    byte_length: int
    sample_rate: int | None = None
    channels: int | None = None


def inspect_audio(audio: bytes) -> AudioMetadata:
    wav_metadata = _inspect_wav(audio)
    if wav_metadata is not None:
        return wav_metadata
    return AudioMetadata(
        duration_seconds=max(len(audio) / RAW_FALLBACK_BYTES_PER_SECOND, 0.01),
        format="raw",
        byte_length=len(audio),
    )


def inspect_audio_path(path: Path) -> AudioMetadata:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            channels = wav_file.getnchannels()
    except (EOFError, wave.Error):
        size = path.stat().st_size
        return AudioMetadata(
            duration_seconds=max(size / RAW_FALLBACK_BYTES_PER_SECOND, 0.01),
            format="raw",
            byte_length=size,
        )
    return AudioMetadata(
        duration_seconds=frame_count / frame_rate,
        format="wav",
        byte_length=path.stat().st_size,
        sample_rate=frame_rate,
        channels=channels,
    )


def _inspect_wav(audio: bytes) -> AudioMetadata | None:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            channels = wav_file.getnchannels()
    except (EOFError, wave.Error):
        return None
    if frame_rate <= 0:
        return None
    return AudioMetadata(
        duration_seconds=frame_count / frame_rate,
        format="wav",
        byte_length=len(audio),
        sample_rate=frame_rate,
        channels=channels,
    )
