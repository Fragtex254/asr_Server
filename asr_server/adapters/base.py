from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionTimings:
    total_ms: float = 0.0
    load_ms: float = 0.0
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def to_api(self) -> dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "load_ms": self.load_ms,
            "decode_ms": self.decode_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
        }


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    source_speaker: str | None = None
    speaker_resolution: str | None = None

    def to_api(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
            **({"source_speaker": self.source_speaker} if self.source_speaker is not None else {}),
            **({"speaker_resolution": self.speaker_resolution} if self.speaker_resolution is not None else {}),
        }


@dataclass(frozen=True)
class GenerationMetrics:
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    max_new_tokens: int | None = None
    peak_vram_allocated_mb: float | None = None
    segment_coverage_end_seconds: float | None = None
    segment_coverage_ratio: float | None = None

    def to_api(self) -> dict[str, int | float | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "peak_vram_allocated_mb": self.peak_vram_allocated_mb,
            "segment_coverage_end_seconds": self.segment_coverage_end_seconds,
            "segment_coverage_ratio": self.segment_coverage_ratio,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration: float
    language: str
    warnings: list[str]
    segments: list[TranscriptionSegment] = field(default_factory=list)
    timings: TranscriptionTimings = field(default_factory=TranscriptionTimings)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)


class AsrAdapter(Protocol):
    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        ...

    async def unload(self, cuda_empty_cache: bool) -> None:
        ...

    async def abort(self) -> None:
        """Force-stop an unresponsive worker during bounded service shutdown."""
        ...

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> TranscriptionResult:
        ...

    async def transcribe_batch(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> list[TranscriptionResult]:
        ...
@dataclass(frozen=True)
class AudioPath:
    """A bounded view into a normalized audio file shared with model workers."""

    path: Path
    start: float = 0.0
    end: float | None = None

    @property
    def duration(self) -> float | None:
        if self.end is None:
            return None
        return max(self.end - self.start, 0.0)


@dataclass(frozen=True)
class AudioSilence:
    duration: float


AudioCompositionPart = AudioPath | AudioSilence


@dataclass(frozen=True)
class AudioComposition:
    """A worker-side composition of source slices and synthetic silence.

    ``prefix_duration`` identifies the replay prefix so callers can remove it
    from model timestamps without inspecting the materialized temporary file.
    """

    parts: tuple[AudioCompositionPart, ...]
    prefix_duration: float

    @property
    def duration(self) -> float:
        return sum(
            part.duration if isinstance(part, AudioSilence) else (part.duration or 0.0)
            for part in self.parts
        )


AudioInput = bytes | AudioPath | AudioComposition
