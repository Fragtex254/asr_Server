# WSL ASR Server

This repository contains the planning documents and agent prompts for a local-network ASR gateway that will run on Windows WSL Arch Linux and be called from Mac mini projects.

## Layout

- `docs/asr-server-prd.md`: product requirements and API contract.
- `prompts/server-agent.md`: implementation instructions for the WSL Arch Linux service-side agent.
- `prompts/request-client-agent.md`: integration instructions for Mac-side client projects.
- `asr_server/`: FastAPI app, model registry, lifecycle manager, and mock ASR adapter.
- `tests/`: API and lifecycle behavior tests that run without CUDA.
- `scripts/asr_client.py`: Mac-side validation client that bypasses local proxy settings.

## Deployment Target

The service is intended to run inside WSL Arch Linux at:

```text
/home/fragt/services/asr-server
```

The public LAN API endpoint is:

```text
http://192.168.31.137:18080
```

Mac-side requests to this LAN endpoint must bypass local proxies, for example:

```bash
curl --noproxy '*' http://192.168.31.137:18080/health
```

## Current Status

This repository now contains the FastAPI service skeleton, mock ASR adapter, lifecycle manager, and API tests that can run on macOS without CUDA or model downloads. Real Qwen3-ASR model dependencies, CUDA validation, and deployment remain WSL Arch Linux work.

## Local Development

Use uv as the source of truth for Python and dependencies:

```bash
uv sync
uv run pytest -q
uv run mypy asr_server tests scripts
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

Python is pinned to 3.12 via `.python-version` and `pyproject.toml`; exact Python package versions are locked in `uv.lock`.

If a machine must use conda first, create only the outer Python/uv environment with:

```bash
conda env create -f environment.yml
conda activate asr-server
uv sync
```

Do not install CUDA, Qwen model packages, or model caches on the Mac mini.

## Mac-Side Validation Client

The helper client disables environment proxy use with `httpx.Client(trust_env=False)`.

```bash
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 check
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 transcribe /path/to/audio.wav --model qwen3-asr-1.7b --backend auto
```
