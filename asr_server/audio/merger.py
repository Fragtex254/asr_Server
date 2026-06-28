from __future__ import annotations

from dataclasses import dataclass

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings
from asr_server.audio.splitter import AudioChunk
from asr_server.audio.transcript import TranscriptDocument, build_transcript_document


@dataclass(frozen=True)
class MergedTranscription:
    text: str
    language: str
    duration: float
    warnings: list[str]
    timings: TranscriptionTimings
    chunks: list[dict[str, object]]


def merge_transcription_results(
    chunks: list[AudioChunk],
    results: list[TranscriptionResult],
    *,
    source_duration: float,
    preserve_segments: bool,
    timings: TranscriptionTimings,
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
    warnings = _unique_warnings(results)
    language = next((result.language for result in results if result.language), "auto")
    return MergedTranscription(
        text=document.text,
        language=language,
        duration=source_duration,
        warnings=warnings,
        timings=timings,
        chunks=_chunk_payload(chunks, results, document) if preserve_segments else [],
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
    chunks: list[AudioChunk],
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
            }
        )
    return payload
