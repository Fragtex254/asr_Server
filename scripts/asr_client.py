from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://192.168.31.137:18080"
DEFAULT_MODEL = "qwen3-asr-1.7b"
FALLBACK_MODEL = "qwen3-asr-0.6b"


def make_client(timeout_seconds: float) -> httpx.Client:
    timeout = httpx.Timeout(connect=5.0, read=timeout_seconds, write=30.0, pool=5.0)
    return httpx.Client(timeout=timeout, trust_env=False)


def response_error_code(response: httpx.Response) -> str | None:
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def discover_models(client: httpx.Client, base_url: str) -> list[dict[str, Any]]:
    response = client.get(f"{base_url}/v1/models")
    response.raise_for_status()
    body = response.json()
    models = body.get("models")
    if not isinstance(models, list):
        raise RuntimeError("/v1/models response did not contain a models list")
    return [model for model in models if isinstance(model, dict)]


def choose_fallback_model(models: list[dict[str, Any]], current_model: str) -> str | None:
    for preferred in (FALLBACK_MODEL, DEFAULT_MODEL):
        if preferred != current_model and any(model.get("id") == preferred for model in models):
            return preferred
    for model in models:
        model_id = model.get("id")
        capabilities = model.get("capabilities")
        if (
            isinstance(model_id, str)
            and model_id != current_model
            and isinstance(capabilities, dict)
            and capabilities.get("transcription") is True
        ):
            return model_id
    return None


def declared_backends(models: list[dict[str, Any]], model_id: str) -> list[str]:
    for model in models:
        if model.get("id") != model_id:
            continue
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, dict):
            return []
        backends = capabilities.get("backends")
        if not isinstance(backends, list):
            return []
        return [backend for backend in backends if isinstance(backend, str)]
    return []


def choose_backend(models: list[dict[str, Any]], model_id: str, requested_backend: str) -> str:
    if requested_backend == "auto":
        return "auto"
    backends = declared_backends(models, model_id)
    if requested_backend in backends:
        return requested_backend
    if backends:
        return backends[0]
    return "auto"


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


def post_transcription(
    client: httpx.Client,
    base_url: str,
    audio_path: Path,
    *,
    model: str,
    backend: str,
    language: str,
    timestamps: str,
) -> httpx.Response:
    with audio_path.open("rb") as audio_file:
        return client.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, audio_file, "application/octet-stream")},
            data={
                "model": model,
                "backend": backend,
                "language": language,
                "response_format": "json",
                "timestamps": timestamps,
            },
        )


def transcribe(
    base_url: str,
    audio_path: Path,
    model: str,
    backend: str,
    language: str,
    timestamps: str,
) -> dict[str, Any]:
    with make_client(timeout_seconds=1800.0) as client:
        models = discover_models(client, base_url)
        current_model = model
        current_backend = choose_backend(models, current_model, backend)
        current_timestamps = timestamps
        capability_downgraded = False

        for attempt in range(1, 22):
            response = post_transcription(
                client,
                base_url,
                audio_path,
                model=current_model,
                backend=current_backend,
                language=language,
                timestamps=current_timestamps,
            )
            if response.status_code == 200:
                result = response.json()
                if not isinstance(result, dict):
                    raise RuntimeError("transcription response was not a JSON object")
                return result

            code = response_error_code(response)
            if response.status_code == 409 and code == "model_loading" and attempt <= 20:
                time.sleep(3)
                continue
            if response.status_code == 409 and code == "model_unloading_scheduled":
                fallback = choose_fallback_model(models, current_model)
                if fallback is not None:
                    current_model = fallback
                    current_backend = choose_backend(models, current_model, backend)
                    continue
            if response.status_code == 422 and code == "capability_not_supported" and not capability_downgraded:
                current_timestamps = "none"
                current_backend = choose_backend(models, current_model, "auto")
                capability_downgraded = True
                continue
            response.raise_for_status()

    raise RuntimeError("transcription retries exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac 侧 ASR 服务验证客户端。")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="检查 /health 和 /v1/models，且禁用环境代理。")

    transcribe_parser = subparsers.add_parser("transcribe", help="上传一个音频文件进行转录。")
    transcribe_parser.add_argument("audio_path", type=Path)
    transcribe_parser.add_argument("--model", default=DEFAULT_MODEL)
    transcribe_parser.add_argument("--backend", default="auto", choices=["auto", "transformers"])
    transcribe_parser.add_argument("--language", default="auto")
    transcribe_parser.add_argument("--timestamps", default="none", choices=["none", "word", "char"])

    args = parser.parse_args()
    if args.command == "check":
        check_server(args.base_url)
        return

    result = transcribe(args.base_url, args.audio_path, args.model, args.backend, args.language, args.timestamps)
    print(result)


if __name__ == "__main__":
    main()
