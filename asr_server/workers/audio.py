from __future__ import annotations

import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from asr_server.adapters.base import AudioInput, AudioPath


@contextmanager
def materialized_audio_path(audio: AudioInput, *, suffix: str = ".wav") -> Iterator[Path]:
    """Yield a model-readable path while bounding memory to one current chunk."""

    if isinstance(audio, AudioPath) and audio.end is None and audio.start <= 0:
        yield audio.path
        return

    directory = audio.path.parent if isinstance(audio, AudioPath) else None
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix="worker_chunk_", dir=directory, delete=False) as output:
        output_path = Path(output.name)
        if isinstance(audio, bytes):
            output.write(audio)
        else:
            with wave.open(str(audio.path), "rb") as source:
                frame_rate = source.getframerate()
                params = source.getparams()
                start_frame = max(int(audio.start * frame_rate), 0)
                end_seconds = audio.end if audio.end is not None else source.getnframes() / frame_rate
                end_frame = min(int(end_seconds * frame_rate), source.getnframes())
                source.setpos(min(start_frame, source.getnframes()))
                frames = source.readframes(max(end_frame - start_frame, 0))
            with wave.open(str(output_path), "wb") as target:
                target.setparams(params)
                target.writeframes(frames)
    try:
        yield output_path
    finally:
        output_path.unlink(missing_ok=True)


def audio_duration_seconds(audio: AudioInput) -> float:
    if isinstance(audio, AudioPath):
        if audio.duration is not None:
            return max(audio.duration, 0.01)
        with wave.open(str(audio.path), "rb") as source:
            return max(source.getnframes() / source.getframerate(), 0.01)
    # Retained only for the compatibility/test path. Normalized PCM is 16-bit,
    # mono, 16 kHz, so WAV payloads are approximately 32,000 bytes per second.
    try:
        with wave.open(__import__("io").BytesIO(audio), "rb") as source:
            return max(source.getnframes() / source.getframerate(), 0.01)
    except (EOFError, wave.Error):
        return max(len(audio) / 16_000, 0.01)
