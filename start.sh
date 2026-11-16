#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .venv/bin/python ]]; then
  echo "Missing server environment. Run ./install.sh first." >&2
  exit 1
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
HOST="${TTS_HOST:-127.0.0.1}"
PORT="${TTS_PORT:-8025}"
if [[ "${1:-}" == "--lan" ]]; then
  HOST=0.0.0.0
  if [[ -z "${TTS_API_KEY:-}" ]]; then
    echo "LAN mode requires TTS_API_KEY in .env" >&2
    exit 1
  fi
fi
exec .venv/bin/python -m uvicorn tts_api.server:app --host "$HOST" --port "$PORT"
