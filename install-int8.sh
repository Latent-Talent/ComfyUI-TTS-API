#!/usr/bin/env bash
# Opt-in TorchAO W8A8 support. Does not replace Torch, CUDA or Python.
set -euo pipefail
cd "$(dirname "$0")"
target="${1:-core}"
if [[ "$target" != "core" && "$target" != "index" && "$target" != "both" ]]; then
  echo "Usage: bash install-int8.sh [core|index|both]" >&2
  exit 2
fi

choose_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return; fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then echo "$HOME/.local/bin/uv"; return; fi
  if [[ -x "$HOME/.cargo/bin/uv" ]]; then echo "$HOME/.cargo/bin/uv"; return; fi
  local tools_dir="${TTS_MODELS_DIR:-$(cd ../.. && pwd)/models/TTS}/IndexTTS-2.5/tools/uv"
  if [[ -x "$tools_dir" ]]; then echo "$tools_dir"; return; fi
  return 1
}

install_one() {
  local python="$1"
  if [[ ! -x "$python" ]]; then
    echo "Missing Python environment: $python" >&2
    return 1
  fi
  # Intentionally do not install dependencies; doing so can overwrite the
  # worker's known-good torch/torchaudio/CUDA wheels.
  if "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m pip install --no-deps torchao
  else
    local uv
    if ! uv="$(choose_uv)"; then
      echo "uv or pip is required to install torchao in $python" >&2
      return 1
    fi
    "$uv" pip install --python "$python" --no-deps torchao
  fi
  "$python" - <<'PY'
import torch, torchao
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
print("Torch:", torch.__version__, "TorchAO:", getattr(torchao, "__version__", "installed"))
print("CUDA available:", torch.cuda.is_available())
print("INT8 W8A8 operators imported successfully")
PY
}

if [[ "$target" == "core" || "$target" == "both" ]]; then
  install_one "$(pwd)/.venv/bin/python"
fi
if [[ "$target" == "index" || "$target" == "both" ]]; then
  models_dir="${TTS_MODELS_DIR:-$(cd ../.. && pwd)/models/TTS}"
  install_one "$models_dir/IndexTTS-2.5/source/.venv/bin/python"
fi
echo "TorchAO installed. Use Server → Model-specific defaults to select INT8."
echo "Conversion occurs in the selected TTS worker when the model next loads."
