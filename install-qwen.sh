#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "TTS environment missing. Run 'bash install.sh' first." >&2
  exit 1
fi

# Install the optional Qwen inference package into the *existing isolated TTS
# service venv*. Do not change Python, Torch, or packages in ComfyUI's venv.
.venv/bin/python -m pip install 'qwen-tts>=0.1.1,<1'
.venv/bin/python -m pip check
.venv/bin/python -c 'from qwen_tts import Qwen3TTSModel; print("Qwen3-TTS Python backend ready")'
echo "Restart the managed TTS API from the ComfyUI sidebar to activate Qwen3-TTS."
