from __future__ import annotations

import io
import wave
from array import array
from pathlib import Path

import pytest

from asr_server.audio import splitter as splitter_module
from asr_server.audio.preprocess import normalize_audio_to_wav
from asr_server.audio.splitter import split_audio
from asr_server.errors import AsrError


LONG_MP3_FIXTURE = Path("test-fixtures/audio/test_long.mp3")


def make_wav(duration_seconds: float, sample_rate: int = 8_000) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return output.getvalue()


def make_segmented_wav(segments: list[tuple[float, int]], sample_rate: int = 16_000) -> bytes:
    samples = array("h")
    for duration_seconds, amplitude in segments:
        samples.extend([amplitude] * int(duration_seconds * sample_rate))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


def wav_duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def test_short_audio_does_not_split_in_auto_mode() -> None:
    result = split_audio(
        b"a" * 160,
        split_strategy="auto",
        max_chunk_seconds=1.0,
        overlap_seconds=0.1,
    )

    assert result.strategy == "none"
    assert len(result.chunks) == 1
    assert result.chunks[0].start == 0.0
    assert result.chunks[0].end == pytest.approx(0.01)


def test_fixed_strategy_splits_raw_audio_with_overlap() -> None:
    result = split_audio(
        b"a" * 1760,
        split_strategy="fixed",
        max_chunk_seconds=0.04,
        overlap_seconds=0.01,
    )

    assert result.strategy == "fixed"
    assert len(result.chunks) == 4
    assert result.chunks[0].start == 0.0
    assert result.chunks[1].start == pytest.approx(0.03)
    assert result.chunks[-1].end == pytest.approx(0.11)
    assert all(current.start < current.end for current in result.chunks)
    assert all(
        previous.start <= current.start <= previous.end
        for previous, current in zip(result.chunks, result.chunks[1:])
    )


def test_fixed_strategy_keeps_wav_chunks_decodable() -> None:
    result = split_audio(
        make_wav(0.11),
        split_strategy="fixed",
        max_chunk_seconds=0.04,
        overlap_seconds=0.01,
    )

    assert len(result.chunks) == 4
    assert wav_duration(result.chunks[0].audio) == pytest.approx(0.04)
    assert wav_duration(result.chunks[-1].audio) == pytest.approx(0.02)


def test_energy_strategy_splits_wav_on_silence() -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
            (0.2, 0),
        ]
    )

    result = split_audio(
        audio,
        split_strategy="energy",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.vad_backend == "energy"
    assert len(result.chunks) == 2
    assert result.chunks[0].start < 0.22
    assert result.chunks[0].end > 0.55
    assert result.chunks[1].start > 1.0
    assert result.chunks[1].end < 1.6


def test_vad_alias_falls_back_to_energy_when_silero_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
        ]
    )

    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    result = split_audio(
        audio,
        split_strategy="vad",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.requested_strategy == "vad"
    assert result.vad_backend == "energy"
    assert any(warning.startswith("silero_vad_unavailable") for warning in result.warnings)
    assert len(result.chunks) == 2


def test_auto_prefers_silero_for_long_audio_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
        ]
    )

    def silero_intervals(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        return [(0.17, 0.63), (1.07, 1.5)]

    monkeypatch.setattr(splitter_module, "_silero_intervals", silero_intervals)

    result = split_audio(
        audio,
        split_strategy="auto",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "silero"
    assert result.requested_strategy == "auto"
    assert result.vad_backend == "silero"
    assert len(result.chunks) == 2


def test_auto_falls_back_to_energy_when_silero_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
        ]
    )

    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    result = split_audio(
        audio,
        split_strategy="auto",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.requested_strategy == "auto"
    assert result.vad_backend == "energy"
    assert len(result.chunks) == 2
    assert any(warning.startswith("silero_vad_unavailable") for warning in result.warnings)


def test_overlap_must_be_smaller_than_chunk_length() -> None:
    with pytest.raises(AsrError) as exc_info:
        split_audio(
            b"a" * 1600,
            split_strategy="fixed",
            max_chunk_seconds=1.0,
            overlap_seconds=1.0,
        )

    assert exc_info.value.code == "bad_request"


def test_duration_limit_is_enforced_before_sync_transcription() -> None:
    with pytest.raises(AsrError) as exc_info:
        split_audio(
            b"a" * int(7201 * 16_000),
            split_strategy="none",
            max_chunk_seconds=None,
            overlap_seconds=None,
        )

    assert exc_info.value.code == "duration_limit_exceeded"


def test_long_mp3_fixture_is_preprocessed_and_energy_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    normalized = normalize_audio_to_wav(LONG_MP3_FIXTURE.read_bytes())
    result = split_audio(
        normalized.audio,
        split_strategy="auto",
        max_chunk_seconds=180.0,
        overlap_seconds=2.0,
    )

    assert result.metadata.format == "wav"
    assert result.metadata.sample_rate == 16_000
    assert result.metadata.channels == 1
    assert result.metadata.duration_seconds == pytest.approx(3630.0, abs=2.0)
    assert result.strategy == "energy"
    assert result.vad_backend == "energy"
    assert len(result.chunks) > 1
    assert all(chunk.duration <= 180.0 for chunk in result.chunks)
    assert all(chunk.start < chunk.end for chunk in result.chunks)
