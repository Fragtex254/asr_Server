from __future__ import annotations

import io
import math
import wave
from array import array
from dataclasses import dataclass
from typing import Literal, cast

from asr_server.audio.metadata import AudioMetadata, inspect_audio
from asr_server.errors import AsrError


SplitStrategy = Literal["auto", "none", "fixed", "vad"]

DEFAULT_SOFT_CHUNK_SECONDS = 180.0
DEFAULT_HARD_CHUNK_SECONDS = 300.0
DEFAULT_OVERLAP_SECONDS = 2.0
MAX_AUDIO_SECONDS_PER_FILE = 7200.0
VAD_FRAME_SECONDS = 0.03
VAD_MIN_SPEECH_SECONDS = 0.15
VAD_MIN_SILENCE_SECONDS = 0.3
VAD_RELATIVE_THRESHOLD = 0.10
VAD_ABSOLUTE_THRESHOLD = 300.0


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float
    audio: bytes

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass(frozen=True)
class SplitResult:
    strategy: str
    requested_strategy: str
    chunks: list[AudioChunk]
    metadata: AudioMetadata
    soft_chunk_seconds: float
    hard_chunk_seconds: float
    overlap_seconds: float

    def summary(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "requested_strategy": self.requested_strategy,
            "chunk_count": len(self.chunks),
            "soft_chunk_seconds": self.soft_chunk_seconds,
            "hard_chunk_seconds": self.hard_chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
        }


def split_audio(
    audio: bytes,
    *,
    split_strategy: str,
    max_chunk_seconds: float | None,
    overlap_seconds: float | None,
) -> SplitResult:
    strategy = _parse_strategy(split_strategy)
    metadata = inspect_audio(audio)
    if metadata.duration_seconds > MAX_AUDIO_SECONDS_PER_FILE:
        raise AsrError(
            422,
            "duration_limit_exceeded",
            "audio is longer than the server file duration limit",
            {"duration_seconds": metadata.duration_seconds, "max_audio_seconds_per_file": MAX_AUDIO_SECONDS_PER_FILE},
        )

    chunk_seconds = max_chunk_seconds or DEFAULT_SOFT_CHUNK_SECONDS
    overlap = min(DEFAULT_OVERLAP_SECONDS, chunk_seconds / 10) if overlap_seconds is None else overlap_seconds
    _validate_chunk_options(chunk_seconds, overlap)

    if strategy == "none":
        return _single_chunk(audio, metadata, requested_strategy=strategy, overlap_seconds=overlap)

    if strategy in {"auto", "vad"}:
        vad_chunks = _split_vad(audio, metadata, chunk_seconds, overlap)
        if vad_chunks:
            return SplitResult(
                strategy="vad",
                requested_strategy=strategy,
                chunks=vad_chunks,
                metadata=metadata,
                soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
                hard_chunk_seconds=DEFAULT_HARD_CHUNK_SECONDS,
                overlap_seconds=overlap,
            )
        if strategy == "vad":
            return _single_chunk(
                audio,
                metadata,
                requested_strategy=strategy,
                resolved_strategy="vad",
                overlap_seconds=overlap,
            )

    if strategy == "auto" and metadata.duration_seconds <= chunk_seconds:
        return _single_chunk(audio, metadata, requested_strategy=strategy, overlap_seconds=overlap)

    chunks = _split_wav(audio, metadata, chunk_seconds, overlap)
    if chunks is None:
        chunks = _split_raw(audio, metadata, chunk_seconds, overlap)
    return SplitResult(
        strategy="fixed",
        requested_strategy=strategy,
        chunks=chunks,
        metadata=metadata,
        soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
        hard_chunk_seconds=DEFAULT_HARD_CHUNK_SECONDS,
        overlap_seconds=overlap,
    )


def _parse_strategy(value: str) -> SplitStrategy:
    if value in {"auto", "none", "fixed", "vad"}:
        return cast(SplitStrategy, value)
    raise AsrError(
        400,
        "bad_request",
        f"unsupported split_strategy: {value}",
        {"supported": ["auto", "none", "fixed", "vad"]},
    )


def _validate_chunk_options(max_chunk_seconds: float, overlap_seconds: float) -> None:
    if max_chunk_seconds <= 0:
        raise AsrError(400, "bad_request", "max_chunk_seconds must be greater than 0")
    if max_chunk_seconds > DEFAULT_HARD_CHUNK_SECONDS:
        raise AsrError(
            422,
            "duration_limit_exceeded",
            "max_chunk_seconds exceeds the server hard chunk limit",
            {"max_chunk_seconds": max_chunk_seconds, "hard_chunk_seconds": DEFAULT_HARD_CHUNK_SECONDS},
        )
    if overlap_seconds < 0:
        raise AsrError(400, "bad_request", "overlap_seconds must be greater than or equal to 0")
    if overlap_seconds >= max_chunk_seconds:
        raise AsrError(
            400,
            "bad_request",
            "overlap_seconds must be smaller than max_chunk_seconds",
        )


def _single_chunk(
    audio: bytes,
    metadata: AudioMetadata,
    *,
    requested_strategy: str,
    overlap_seconds: float,
    resolved_strategy: str = "none",
) -> SplitResult:
    return SplitResult(
        strategy=resolved_strategy,
        requested_strategy=requested_strategy,
        chunks=[AudioChunk(index=0, start=0.0, end=metadata.duration_seconds, audio=audio)],
        metadata=metadata,
        soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
        hard_chunk_seconds=DEFAULT_HARD_CHUNK_SECONDS,
        overlap_seconds=overlap_seconds,
    )


def _chunk_windows(duration_seconds: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        windows.append((start, end))
        if end >= duration_seconds:
            break
        start = end - overlap_seconds
    return windows


def _interval_windows(
    start: float,
    end: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + chunk_seconds, end)
        windows.append((cursor, window_end))
        if window_end >= end:
            break
        cursor = window_end - overlap_seconds
    return windows


def _split_vad(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk]:
    intervals = _vad_intervals(audio, metadata, padding_seconds=overlap_seconds)
    if not intervals:
        return []
    windows = _pack_vad_intervals(intervals, chunk_seconds, overlap_seconds)
    return _slice_wav_windows(audio, metadata, windows) or []


def _vad_intervals(
    audio: bytes,
    metadata: AudioMetadata,
    *,
    padding_seconds: float,
) -> list[tuple[float, float]]:
    if metadata.format != "wav":
        return []
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() != 1:
                return []
            frame_rate = source.getframerate()
            pcm = source.readframes(source.getnframes())
    except (EOFError, wave.Error):
        return []

    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return []

    frame_samples = max(int(frame_rate * VAD_FRAME_SECONDS), 1)
    rms_values = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if not frame:
            continue
        rms_values.append(math.sqrt(sum(sample * sample for sample in frame) / len(frame)))
    if not rms_values:
        return []

    threshold = max(VAD_ABSOLUTE_THRESHOLD, max(rms_values) * VAD_RELATIVE_THRESHOLD)
    speech_frames = [rms >= threshold for rms in rms_values]
    raw_intervals: list[tuple[float, float]] = []
    start_frame: int | None = None
    for index, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = index
        if not is_speech and start_frame is not None:
            raw_intervals.append((start_frame * VAD_FRAME_SECONDS, index * VAD_FRAME_SECONDS))
            start_frame = None
    if start_frame is not None:
        raw_intervals.append((start_frame * VAD_FRAME_SECONDS, len(speech_frames) * VAD_FRAME_SECONDS))

    merged = _merge_vad_intervals(raw_intervals)
    padded = []
    for start, end in merged:
        if end - start < VAD_MIN_SPEECH_SECONDS:
            continue
        padded.append((max(start - padding_seconds, 0.0), min(end + padding_seconds, metadata.duration_seconds)))
    return padded


def _merge_vad_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start - merged[-1][1] > VAD_MIN_SILENCE_SECONDS:
            merged.append((start, end))
            continue
        previous_start, _previous_end = merged[-1]
        merged[-1] = (previous_start, end)
    return merged


def _pack_vad_intervals(
    intervals: list[tuple[float, float]],
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None
    for interval_start, interval_end in intervals:
        if interval_end - interval_start > chunk_seconds:
            if current_start is not None and current_end is not None:
                windows.append((current_start, current_end))
                current_start = None
                current_end = None
            windows.extend(_interval_windows(interval_start, interval_end, chunk_seconds, overlap_seconds))
            continue
        if current_start is None or current_end is None:
            current_start = interval_start
            current_end = interval_end
            continue
        if interval_end - current_start <= chunk_seconds:
            current_end = interval_end
            continue
        windows.append((current_start, current_end))
        current_start = max(interval_start - overlap_seconds, 0.0)
        current_end = interval_end
    if current_start is not None and current_end is not None:
        windows.append((current_start, current_end))
    return windows


def _split_raw(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk]:
    bytes_per_second = len(audio) / metadata.duration_seconds
    chunks = []
    for index, (start, end) in enumerate(_chunk_windows(metadata.duration_seconds, chunk_seconds, overlap_seconds)):
        start_byte = int(start * bytes_per_second)
        end_byte = max(start_byte + 1, int(end * bytes_per_second))
        chunks.append(AudioChunk(index=index, start=start, end=end, audio=audio[start_byte:end_byte]))
    return chunks


def _split_wav(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk] | None:
    if metadata.format != "wav":
        return None
    return _slice_wav_windows(audio, metadata, _chunk_windows(metadata.duration_seconds, chunk_seconds, overlap_seconds))


def _slice_wav_windows(
    audio: bytes,
    metadata: AudioMetadata,
    windows: list[tuple[float, float]],
) -> list[AudioChunk] | None:
    if metadata.format != "wav":
        return None
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            frame_rate = source.getframerate()
            params = source.getparams()
            chunks = []
            for index, (start, end) in enumerate(windows):
                start_frame = int(start * frame_rate)
                end_frame = min(int(end * frame_rate), source.getnframes())
                source.setpos(start_frame)
                frames = source.readframes(max(end_frame - start_frame, 0))
                output = io.BytesIO()
                with wave.open(output, "wb") as target:
                    target.setparams(params)
                    target.writeframes(frames)
                chunks.append(AudioChunk(index=index, start=start, end=end, audio=output.getvalue()))
    except (EOFError, wave.Error):
        return None
    return chunks
