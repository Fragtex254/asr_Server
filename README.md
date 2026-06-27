# WSL ASR Server

This repository contains the planning documents and agent prompts for a local-network ASR gateway that will run on Windows WSL Arch Linux and be called from Mac mini projects.

## Layout

- `docs/asr-server-prd.md`: product requirements and API contract.
- `prompts/server-agent.md`: implementation instructions for the WSL Arch Linux service-side agent.
- `prompts/request-client-agent.md`: integration instructions for Mac-side client projects.

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

This repository currently contains the project specification and agent handoff prompts. The FastAPI implementation should be created against `docs/asr-server-prd.md`, with lightweight development possible on macOS and GPU/model deployment performed inside WSL Arch Linux.
