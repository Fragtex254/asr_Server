from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_server.audio.transcript import build_transcript_document, write_transcript_artifacts


CHUNK_HEADING_RE = re.compile(
    r"^### Chunk (?P<index>\d+) \[(?P<start>\d\d:\d\d:\d\d\.\d{3}) - (?P<end>\d\d:\d\d:\d\d\.\d{3})\]$"
)


def parse_time(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_metadata(lines: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in lines:
        if not line.startswith("- "):
            continue
        key, separator, value = line[2:].partition(": ")
        if separator:
            metadata[key.strip()] = value.strip()
    return metadata


def parse_markdown(markdown_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    metadata = parse_metadata(lines)
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = CHUNK_HEADING_RE.match(lines[index])
        if match is None:
            index += 1
            continue

        start = parse_time(match.group("start"))
        end = parse_time(match.group("end"))
        language = "auto"
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        while index < len(lines) and lines[index].startswith("- "):
            key, separator, value = lines[index][2:].partition(": ")
            if separator and key == "language":
                language = value.strip()
            index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1

        text_lines = []
        while index < len(lines) and not lines[index].startswith("### Chunk ") and lines[index] != "## Full Text":
            text_lines.append(lines[index])
            index += 1
        text = "\n".join(text_lines).strip()
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "language": language,
            }
        )
    metadata["source_markdown"] = str(markdown_path)
    return metadata, segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export transcript JSON/TXT/SRT artifacts from segmented markdown.")
    parser.add_argument("markdown_path", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix or args.markdown_path.with_suffix("")
    metadata, raw_segments = parse_markdown(args.markdown_path)
    document = build_transcript_document(
        raw_segments,
        metadata=metadata,
        timestamp_source="vad_chunk_window",
    )
    paths = write_transcript_artifacts(document, output_prefix)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
