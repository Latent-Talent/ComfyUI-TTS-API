#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.12 is required; set PYTHON to an existing compatible executable." >&2
  exit 1
fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install voxcpm
if [[ ! -f .env ]]; then cp config.example.env .env; fi
echo "Install complete. Restart ComfyUI normally; it starts and stops the TTS API automatically."
