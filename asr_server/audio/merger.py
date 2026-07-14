from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings
from asr_server.audio.transcript import TranscriptDocument, build_transcript_document


class ChunkLike(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def duration(self) -> float: ...


@dataclass(frozen=True)
class MergedTranscription:
    text: str
    language: str
    duration: float
    warnings: list[str]
    timings: TranscriptionTimings
    chunks: list[dict[str, object]]
    segments: list[dict[str, object]]


def merge_transcription_results(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
    *,
    source_duration: float,
    preserve_segments: bool,
    timings: TranscriptionTimings,
    speaker_scope: Literal["global", "chunk"] = "chunk",
) -> MergedTranscription:
    raw_segments = [
        {
            "start": chunk.start,
            "end": chunk.end,
            "text": result.text,
            "language": result.language,
        }
        for chunk, result in zip(chunks, results, strict=True)
    ]
    document = build_transcript_document(
        raw_segments,
        metadata={"source_duration": source_duration},
        timestamp_source="vad_chunk_window",
    )
    segments = _merged_segments(chunks, results, speaker_scope=speaker_scope)
    warnings = _unique_warnings(results)
    if len(chunks) > 1 and segments and "moss_speaker_labels_are_chunk_local" not in warnings:
        warnings = [*warnings, "moss_speaker_labels_are_chunk_local"]
    language = next((result.language for result in results if result.language), "auto")
    merged_text = document.text
    if segments and all(result.segments for result in results):
        merged_text = "\n".join(str(segment["text"]) for segment in segments if str(segment["text"]).strip())
    return MergedTranscription(
        text=merged_text,
        language=language,
        duration=source_duration,
        warnings=warnings,
        timings=timings,
        chunks=_chunk_payload(chunks, results, document) if preserve_segments else [],
        segments=segments,
    )


def _unique_warnings(results: list[TranscriptionResult]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for result in results:
        for warning in result.warnings:
            if warning not in seen:
                warnings.append(warning)
                seen.add(warning)
    return warnings


def _chunk_payload(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
    document: TranscriptDocument,
) -> list[dict[str, object]]:
    payload = []
    for chunk, result, segment in zip(chunks, results, document.segments, strict=True):
        payload.append(
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "duration": chunk.duration,
                "text": segment.text,
                "raw_text": result.text,
                "language": result.language,
                "timestamp_source": segment.timestamp_source,
                "overlap_seconds": segment.overlap_seconds,
                "deduped_prefix_chars": segment.deduped_prefix_chars,
                "warnings": result.warnings,
                "timings": result.timings.to_api(),
                "segments": [
                    {
                        "start": chunk.start + item.start,
                        "end": chunk.start + item.end,
                        "speaker": item.speaker,
                        "text": item.text,
                    }
                    for item in result.segments
                ],
            }
        )
    return payload


def _merged_segments(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
    *,
    speaker_scope: Literal["global", "chunk"],
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for chunk_position, (chunk, result) in enumerate(zip(chunks, results, strict=True)):
        ownership_start = chunk.start
        ownership_end = chunk.end
        if chunk_position > 0:
            previous = chunks[chunk_position - 1]
            if previous.end > chunk.start:
                ownership_start = (previous.end + chunk.start) / 2
        if chunk_position + 1 < len(chunks):
            following = chunks[chunk_position + 1]
            if following.start < chunk.end:
                ownership_end = (following.start + chunk.end) / 2
        for segment in result.segments:
            absolute_start = chunk.start + segment.start
            absolute_end = chunk.start + segment.end
            midpoint = (absolute_start + absolute_end) / 2
            if midpoint < ownership_start or midpoint > ownership_end:
                continue
            speaker = segment.speaker
            scoped_speaker = (
                speaker
                if speaker_scope == "global"
                else f"chunk-{chunk.index:04d}:{speaker}" if speaker is not None else None
            )
            segments.append(
                {
                    "start": absolute_start,
                    "end": absolute_end,
                    "speaker": scoped_speaker,
                    "speaker_label": speaker,
                    "speaker_scope": speaker_scope,
                    "chunk_index": chunk.index,
                    "text": segment.text,
                }
            )
    return segments
