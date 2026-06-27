# Agent Development Guide

## Project Context

This repository defines and will host a local-network ASR gateway for Mac mini clients calling a GPU-backed service running inside Windows WSL Arch Linux.

Primary deployment target:

```text
/home/fragt/services/asr-server
```

Public LAN endpoint:

```text
http://192.168.31.137:18080
```

The Mac mini is only a lightweight development and client-validation machine. GPU inference, CUDA validation, model package installation, and long-running service deployment must happen inside WSL Arch Linux on the Windows PC.

## Required Reading

Before implementing service behavior, read:

```text
docs/asr-server-prd.md
```

Use these prompts when handing work to focused agents:

```text
prompts/server-agent.md
prompts/request-client-agent.md
```

## Recommended Installed Skills

The following Matt Pocock skills are useful for this project and may be invoked when they fit the task:

- `grill-me`: stress-test an unclear plan before implementation.
- `grilling`: ask one focused design question at a time until requirements are sharp.
- `grill-with-docs`: refine a plan while also updating durable domain docs and ADRs.
- `setup-matt-pocock-skills`: configure this repo for Matt Pocock's engineering-skill conventions when issue tracking and domain docs are ready.
- `domain-modeling`: define ASR gateway terms, lifecycle states, adapter concepts, and ADRs.
- `codebase-design`: design deep modules and clean adapter/lifecycle seams.
- `diagnosing-bugs`: debug failing tests, lifecycle races, networking failures, and performance regressions.
- `tdd`: build API and lifecycle behavior test-first through public interfaces.
- `implement`: implement a PRD or issue with tests and a final review pass.
- `to-prd`: turn a resolved discussion into a PRD.
- `to-issues`: split a PRD into independently implementable vertical slices.
- `prototype`: create throwaway experiments for lifecycle state machines or adapter behavior.
- `handoff`: summarize a session for another agent without duplicating existing docs.

## Development Principles

- Keep the public API aligned with `docs/asr-server-prd.md`.
- Use Python 3.12, uv, FastAPI, and Uvicorn for the service implementation.
- Keep the service entrypoint listening on `0.0.0.0:18080`.
- Do not use `/mnt/c` as the WSL project location; deploy under `/home/fragt/services/asr-server`.
- Do not expose worker ports `8001` or `8002` to the LAN unless the PRD is explicitly changed.
- Do not reuse historical test port `8765` for implementation, deployment, firewall, or startup configuration.
- Do not add Web UI work before the API, lifecycle manager, tests, and first transcription path are working.

## Cross-Platform Rules

- macOS development may create project skeletons, schemas, tests, mock adapters, and documentation.
- macOS development must not require CUDA, NVIDIA drivers, model downloads, or large local model caches.
- WSL Arch Linux development is responsible for CUDA checks, `nvidia-smi`, disk-space checks, real Qwen/MiMo dependencies, and model inference validation.
- Keep heavy model dependencies lazy-loaded inside adapters so basic imports and tests can run without a GPU.
- Keep path handling POSIX-compatible and avoid hardcoded macOS-only paths in service code.

## API And Lifecycle Rules

- Model capability discovery must come from `GET /v1/models`; clients should not hardcode capabilities.
- Model states must use the PRD enum: `unloaded`, `loading`, `loaded`, `unloading_scheduled`, `unloading`, `error`.
- Every model must maintain active request counting and a lifecycle lock.
- If unload is requested while active requests exist, set `unloading_scheduled`, reject new same-model requests with `409 model_unloading_scheduled`, and unload only after active requests finish.
- Do not force-unload a model while inference is active.
- Return errors using the PRD error envelope:

```json
{
  "error": {
    "code": "model_not_found",
    "message": "unknown model: xxx",
    "details": {}
  }
}
```

## Networking And Security

- The only LAN-facing API port is `18080`.
- Mac clients must bypass local proxies for `192.168.31.137`, for example with `curl --noproxy '*'`.
- Support optional bearer-token authentication, but do not require public-network assumptions.
- Do not add public tunneling, port mapping, or internet exposure.
- Uploaded audio should be stored in temporary locations and cleaned after inference.
- Logs must not save full audio contents by default.

## Testing Expectations

At minimum, cover:

- Health check.
- Model list and model status.
- Model load and unload behavior.
- Unload waiting for active requests.
- Rejection of new requests during `unloading_scheduled`.
- Transcription parameter validation.
- Unsupported capability errors, including MiMo forced alignment.

Use mock adapters first so lifecycle and API behavior can be tested before real model integration.

## Git Hygiene

- Keep generated caches, virtual environments, model caches, uploads, and local runtime data out of Git.
- Keep documentation and prompts in stable paths so agent handoffs remain valid.
- Make focused commits with clear messages.
- Do not commit machine-specific secrets, tokens, downloaded models, or audio samples containing private content.
