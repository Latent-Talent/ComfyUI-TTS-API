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
  # Resolve the release from the worker's installed torch *before* importing
  # torchao. Importing a too-new torchao can break other package imports.
  local spec
  spec="$("$python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from tts_api.torchao_compat import torchao_for_torch
import sys

try:
    torch_version = version("torch")
except PackageNotFoundError:
    print("PyTorch is not installed in this TTS environment.", file=sys.stderr)
    raise SystemExit(1)
try:
    spec = torchao_for_torch(torch_version)
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)
print(spec)
PY
  )"
  echo "Using $spec for $python"
  # Do not replace Torch/CUDA/torchaudio, especially in the locked IndexTTS env.
  # The exact version also replaces an incompatible torchao already installed.
  if "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m pip install --no-deps "$spec"
  else
    local uv
    if ! uv="$(choose_uv)"; then
      echo "uv or pip is required to install torchao in $python" >&2
      return 1
    fi
    UV_LINK_MODE=copy "$uv" pip install --python "$python" --no-deps "$spec"
  fi
  "$python" - <<'PY'
from importlib.metadata import version
from tts_api.torchao_compat import torchao_for_torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
import torch

expected = torchao_for_torch(torch.__version__).partition("==")[2]
installed = version("torchao")
if installed != expected:
    raise RuntimeError(f"Expected TorchAO {expected}, got {installed}")
print("Torch:", torch.__version__, "TorchAO:", installed)
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
