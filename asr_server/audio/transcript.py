from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OVERLAP_SEARCH_CHARS = 300
OVERLAP_PUNCTUATION = set(" \t\r\n，。！？；：、“”‘’（）()[]【】,.!?;:\"'")
LEADING_DEDUPED_PUNCTUATION = " \t\r\n，。！？；：,.!?;:"


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str
    raw_text: str
    language: str
    timestamp_source: str
    overlap_seconds: float
    deduped_prefix_chars: int


@dataclass(frozen=True)
class TranscriptDocument:
    metadata: dict[str, Any]
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n\n".join(segment.text for segment in self.segments if segment.text).strip()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "text": self.text,
            "segments": [asdict(segment) for segment in self.segments],
        }


def build_transcript_document(
    raw_segments: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    timestamp_source: str = "vad_chunk_window",
) -> TranscriptDocument:
    segments: list[TranscriptSegment] = []
    previous_text = ""
    previous_end = 0.0
    for index, raw_segment in enumerate(raw_segments):
        start = float(raw_segment["start"])
        end = float(raw_segment["end"])
        raw_text = str(raw_segment.get("text", "")).strip()
        language = str(raw_segment.get("language", "auto"))
        overlap_seconds = max(previous_end - start, 0.0) if segments else 0.0
        dedupe_chars = _overlap_prefix_length(previous_text, raw_text) if overlap_seconds > 0 else 0
        text = _clean_deduped_text(raw_text[dedupe_chars:])
        segments.append(
            TranscriptSegment(
                index=index,
                start=start,
                end=end,
                text=text,
                raw_text=raw_text,
                language=language,
                timestamp_source=timestamp_source,
                overlap_seconds=overlap_seconds,
                deduped_prefix_chars=dedupe_chars,
            )
        )
        previous_text = text or raw_text
        previous_end = end
    return TranscriptDocument(metadata=metadata, segments=segments)


def write_transcript_artifacts(document: TranscriptDocument, output_prefix: Path) -> dict[str, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.parent / f"{output_prefix.name}.json"
    txt_path = output_prefix.parent / f"{output_prefix.name}.txt"
    srt_path = output_prefix.parent / f"{output_prefix.name}.srt"
    json_path.write_text(
        json.dumps(document.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    txt_path.write_text(document.text + "\n", encoding="utf-8")
    srt_path.write_text(_to_srt(document.segments), encoding="utf-8")
    return {"json": json_path, "txt": txt_path, "srt": srt_path}


def _overlap_prefix_length(previous_text: str, current_text: str) -> int:
    previous = previous_text.strip()
    current = current_text.strip()
    max_length = min(len(previous), len(current), 200)
    for length in range(max_length, 1, -1):
        prefix = current[:length]
        previous_tail = previous[-OVERLAP_SEARCH_CHARS:]
        if previous.endswith(prefix) or prefix in previous_tail:
            return length
        normalized_prefix = _normalize_overlap_text(prefix)
        normalized_tail = _normalize_overlap_text(previous_tail)
        if len(normalized_prefix) >= 2 and (
            normalized_tail.endswith(normalized_prefix) or normalized_prefix in normalized_tail
        ):
            return length
    return 0


def _normalize_overlap_text(text: str) -> str:
    return "".join(character for character in text if character not in OVERLAP_PUNCTUATION)


def _clean_deduped_text(text: str) -> str:
    return text.lstrip(LEADING_DEDUPED_PUNCTUATION).strip()


def _to_srt(segments: list[TranscriptSegment]) -> str:
    blocks: list[str] = []
    for segment in segments:
        if not segment.text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(len(blocks) + 1),
                    f"{_srt_time(segment.start)} --> {_srt_time(segment.end)}",
                    segment.text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _srt_time(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
