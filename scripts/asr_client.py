from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://192.168.31.137:18080"
DEFAULT_MODEL = "qwen3-asr-1.7b"


def make_client(timeout_seconds: float) -> httpx.Client:
    timeout = httpx.Timeout(connect=5.0, read=timeout_seconds, write=30.0, pool=5.0)
    return httpx.Client(timeout=timeout, trust_env=False)


def check_server(base_url: str) -> None:
    with make_client(timeout_seconds=30.0) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        models = client.get(f"{base_url}/v1/models")
        models.raise_for_status()

    print("health:")
    print(health.text)
    print("models:")
    print(models.text)


def transcribe(base_url: str, audio_path: Path, model: str, backend: str, language: str) -> dict[str, Any]:
    with make_client(timeout_seconds=1800.0) as client:
        with audio_path.open("rb") as audio_file:
            response = client.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": (audio_path.name, audio_file, "application/octet-stream")},
                data={
                    "model": model,
                    "backend": backend,
                    "language": language,
                    "response_format": "json",
                    "timestamps": "none",
                },
            )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("transcription response was not a JSON object")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac-side ASR server validation client.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check /health and /v1/models with proxies disabled.")

    transcribe_parser = subparsers.add_parser("transcribe", help="Upload one audio file for transcription.")
    transcribe_parser.add_argument("audio_path", type=Path)
    transcribe_parser.add_argument("--model", default=DEFAULT_MODEL)
    transcribe_parser.add_argument("--backend", default="auto", choices=["auto", "transformers", "vllm"])
    transcribe_parser.add_argument("--language", default="auto")

    args = parser.parse_args()
    if args.command == "check":
        check_server(args.base_url)
        return

    result = transcribe(args.base_url, args.audio_path, args.model, args.backend, args.language)
    print(result)


if __name__ == "__main__":
    main()
