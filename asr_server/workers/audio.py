from __future__ import annotations

import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from asr_server.adapters.base import AudioComposition, AudioInput, AudioPath, AudioSilence


@contextmanager
def materialized_audio_path(audio: AudioInput, *, suffix: str = ".wav") -> Iterator[Path]:
    """Yield a model-readable path while bounding memory to one current chunk."""

    if isinstance(audio, AudioPath) and audio.end is None and audio.start <= 0:
        yield audio.path
        return

    directory = _source_directory(audio)
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix="worker_chunk_", dir=directory, delete=False) as output:
        output_path = Path(output.name)
        if isinstance(audio, bytes):
            output.write(audio)
        elif isinstance(audio, AudioPath):
            _write_composition(output_path, (audio,))
        else:
            _write_composition(output_path, audio.parts)
    try:
        yield output_path
    finally:
        output_path.unlink(missing_ok=True)


def audio_duration_seconds(audio: AudioInput) -> float:
    if isinstance(audio, AudioComposition):
        return max(audio.duration, 0.01)
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


def _source_directory(audio: AudioInput) -> Path | None:
    if isinstance(audio, AudioPath):
        return audio.path.parent
    if isinstance(audio, AudioComposition):
        return next((part.path.parent for part in audio.parts if isinstance(part, AudioPath)), None)
    return None


def _write_composition(output_path: Path, parts: tuple[AudioPath | AudioSilence, ...]) -> None:
    first_source = next((part for part in parts if isinstance(part, AudioPath)), None)
    if first_source is None:
        raise ValueError("audio composition requires at least one source slice")
    with wave.open(str(first_source.path), "rb") as reference:
        params = reference.getparams()
        frame_rate = reference.getframerate()
        frame_width = reference.getnchannels() * reference.getsampwidth()
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        for part in parts:
            if isinstance(part, AudioSilence):
                remaining = max(int(part.duration * frame_rate), 0)
                silence_block = bytes(frame_width * min(remaining, 65_536))
                while remaining:
                    count = min(remaining, 65_536)
                    target.writeframes(silence_block[: count * frame_width])
                    remaining -= count
                continue
            with wave.open(str(part.path), "rb") as source:
                if (
                    source.getnchannels() != params.nchannels
                    or source.getsampwidth() != params.sampwidth
                    or source.getframerate() != params.framerate
                ):
                    raise ValueError("audio composition parts must use the same PCM format")
                start_frame = max(int(part.start * frame_rate), 0)
                end_seconds = part.end if part.end is not None else source.getnframes() / frame_rate
                remaining = max(min(int(end_seconds * frame_rate), source.getnframes()) - start_frame, 0)
                source.setpos(min(start_frame, source.getnframes()))
                while remaining:
                    count = min(remaining, 65_536)
                    frames = source.readframes(count)
                    if not frames:
                        break
                    target.writeframes(frames)
                    remaining -= len(frames) // frame_width
