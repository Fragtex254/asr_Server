#!/usr/bin/env bash
set -euo pipefail

HOST="${ASR_HOST:-0.0.0.0}"
PORT="${ASR_PORT:-18080}"
BASE_URL="${ASR_BASE_URL:-http://127.0.0.1:${PORT}}"
RUN_QWEN_BACKEND_SMOKE="${ASR_RUN_QWEN_BACKEND_SMOKE:-0}"
MODEL_REPO="${ASR_SMOKE_MODEL_REPO:-Qwen/Qwen3-ASR-0.6B}"
AUDIO="${ASR_SMOKE_AUDIO:-test-fixtures/audio/test_short.wav}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

uv run uvicorn asr_server.main:app --host "${HOST}" --port "${PORT}" &
SERVER_PID="$!"

for _ in $(seq 1 60); do
  if curl --noproxy '*' -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl --noproxy '*' -fsS "${BASE_URL}/health" >/dev/null
ASR_BASE_URL="${BASE_URL}" uv run pytest tests/test_http_smoke.py -q

if [[ "${RUN_QWEN_BACKEND_SMOKE}" == "1" ]]; then
  uv run python scripts/qwen_asr_backend_smoke.py \
    --backend transformers \
    --model "${MODEL_REPO}" \
    --audio "${AUDIO}" \
    --language auto
fi
