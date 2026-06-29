from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_server.audio.preprocess import normalize_audio_to_wav
from asr_server.audio.splitter import split_audio
from asr_server.audio.transcript import build_transcript_document, write_transcript_artifacts


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


def discover_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def format_time(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def result_text(result: object) -> str:
    text = getattr(result, "text", "")
    return text if isinstance(text, str) else str(text)


def result_language(result: object) -> str:
    language = getattr(result, "language", "")
    return language if isinstance(language, str) else str(language)


def result_time_stamps(result: object) -> Any:
    return getattr(result, "time_stamps", None)


def assert_cuda_runtime() -> None:
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.__version__ != "2.11.0+cu128":
        raise RuntimeError(f"unexpected torch version: {torch.__version__}; expected 2.11.0+cu128")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"unexpected torch CUDA version: {torch.version.cuda}; expected 12.8")
    if not torch.cuda.is_available():
        raise RuntimeError("torch cannot access CUDA")
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))


def load_model(model_id: str) -> Any:
    print(f"loading model: {model_id}", flush=True)
    qwen_asr: Any = importlib.import_module("qwen_asr")
    model = qwen_asr.Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=4096,
    )
    print("model loaded", flush=True)
    return model


def output_prefix_for(audio_path: Path, output_suffix: str) -> Path:
    return audio_path.with_name(f"{audio_path.stem}{output_suffix}")


def transcribe_audio_file(
    model: Any,
    audio_path: Path,
    *,
    model_id: str,
    output_suffix: str,
    max_chunk_seconds: float,
    overlap_seconds: float,
) -> dict[str, Path]:
    started = perf_counter()
    print(f"\n=== {audio_path} ===", flush=True)
    normalized = normalize_audio_to_wav(audio_path.read_bytes())
    split = split_audio(
        normalized.audio,
        split_strategy="auto",
        max_chunk_seconds=max_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    print(
        f"duration={split.metadata.duration_seconds:.2f}s strategy={split.strategy} chunks={len(split.chunks)}",
        flush=True,
    )

    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audio": str(audio_path),
        "model": model_id,
        "backend": "transformers",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "source_duration_seconds": split.metadata.duration_seconds,
        "split_strategy": split.strategy,
        "requested_strategy": split.requested_strategy,
        "vad_backend": split.vad_backend,
        "split_warnings": split.warnings,
        "chunk_count": len(split.chunks),
        "overlap_seconds": split.overlap_seconds,
        "timestamp_source": "vad_chunk_window",
        "decode_ms": normalized.decode_ms,
    }

    raw_segments: list[dict[str, Any]] = []
    markdown_lines: list[str] = []
    for chunk_index, chunk in enumerate(split.chunks, start=1):
        chunk_started = perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
            chunk_file.write(chunk.audio)
            chunk_path = Path(chunk_file.name)
        try:
            results = model.transcribe(audio=str(chunk_path), language=None)
            first = results[0]
            text = result_text(first).strip()
            language = result_language(first)
            time_stamps = result_time_stamps(first)
        finally:
            chunk_path.unlink(missing_ok=True)
        elapsed = perf_counter() - chunk_started
        raw_segments.append(
            {
                "start": chunk.start,
                "end": chunk.end,
                "text": text,
                "language": language,
                "elapsed_seconds": elapsed,
                "model_time_stamps": time_stamps,
            }
        )
        markdown_lines.extend(
            [
                f"### Chunk {chunk_index:02d} [{format_time(chunk.start)} - {format_time(chunk.end)}]",
                "",
                f"- language: {language}",
                f"- elapsed_seconds: {elapsed:.2f}",
                f"- model_time_stamps: {time_stamps!r}",
                "",
                text,
                "",
            ]
        )
        print(
            f"chunk {chunk_index}/{len(split.chunks)} "
            f"{format_time(chunk.start)}-{format_time(chunk.end)} "
            f"{elapsed:.2f}s chars={len(text)}",
            flush=True,
        )

    document = build_transcript_document(
        raw_segments,
        metadata=metadata,
        timestamp_source="vad_chunk_window",
    )
    prefix = output_prefix_for(audio_path, output_suffix)
    paths = write_transcript_artifacts(document, prefix)
    markdown_path = prefix.parent / f"{prefix.name}.md"
    markdown_path.write_text(
        "\n".join(
            [
                f"# {audio_path.name} Qwen3-ASR 1.7B Transformers Transcript",
                "",
                *[f"- {key}: {value}" for key, value in metadata.items()],
                "",
                "## Segmented Raw Transcript",
                "",
                *markdown_lines,
                "## Deduped Full Text",
                "",
                document.text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["md"] = markdown_path
    print(
        f"done file elapsed={perf_counter() - started:.2f}s "
        f"deduped_segments={sum(1 for segment in document.segments if segment.deduped_prefix_chars > 0)} "
        f"deduped_chars={sum(segment.deduped_prefix_chars for segment in document.segments)}",
        flush=True,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}", flush=True)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcribe audio files with Qwen3-ASR transformers.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output-suffix", default=".qwen3-asr-1_7b")
    parser.add_argument("--max-chunk-seconds", type=float, default=120.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_files = discover_audio_files(args.directory)
    if not audio_files:
        raise RuntimeError(f"no audio files found under {args.directory}")
    print(f"audio files: {len(audio_files)}")
    for path in audio_files:
        print(path)
    assert_cuda_runtime()
    model = load_model(args.model)
    for audio_path in audio_files:
        transcribe_audio_file(
            model,
            audio_path,
            model_id=args.model,
            output_suffix=args.output_suffix,
            max_chunk_seconds=args.max_chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )


if __name__ == "__main__":
    main()
