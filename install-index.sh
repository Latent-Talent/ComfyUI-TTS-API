#!/usr/bin/env bash
# Optional IndexTTS 2.5 setup: own Python 3.11 and Torch; NO changes to
# ComfyUI/.venv, the TTS API's Python 3.12 .venv, or system Python.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
COMFY_ROOT="$(cd ../.. && pwd)"
MODELS_DIR="${TTS_MODELS_DIR:-$COMFY_ROOT/models/TTS}"
INDEX_ROOT="$MODELS_DIR/IndexTTS-2.5"
SOURCE="$INDEX_ROOT/source"
mkdir -p "$INDEX_ROOT/checkpoints" "$MODELS_DIR/huggingface"

# Prefer an existing uv binary (including a previous user-local
# installation). If none is found, install a private standalone copy under
# models/TTS/IndexTTS-2.5/tools. No pip, system Python, sudo, or changes to
# shell startup files are needed.
UV_BIN=""
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$INDEX_ROOT/tools/uv" ]]; then
  UV_BIN="$INDEX_ROOT/tools/uv"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
  UV_BIN="$HOME/.cargo/bin/uv"
else
  echo "uv not found. Installing a private copy for IndexTTS..."
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  if command -v curl >/dev/null 2>&1; then
    curl --fail --silent --show-error --location \
      https://astral.sh/uv/install.sh --output "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet --output-document="$installer" \
      https://astral.sh/uv/install.sh
  else
    echo "Install uv using https://docs.astral.sh/uv/getting-started/installation/ (requires curl or wget)." >&2
    exit 1
  fi
  # Official Astral installer with explicitly scoped, unmanaged destination.
  # UV_UNMANAGED_INSTALL skips changes to shell profiles and PATH.
  env UV_UNMANAGED_INSTALL="$INDEX_ROOT/tools" sh "$installer"
  UV_BIN="$INDEX_ROOT/tools/uv"
  if [[ ! -x "$UV_BIN" ]]; then
    echo "uv standalone installer did not create $UV_BIN" >&2
    exit 1
  fi
fi
echo "Using uv: $UV_BIN"
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch the official IndexTTS source." >&2
  exit 1
fi

if [[ ! -d "$SOURCE/.git" ]]; then
  if [[ -e "$SOURCE" ]]; then
    echo "$SOURCE already exists but is not an IndexTTS git checkout. Refusing overwrite." >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/index-tts/index-tts.git "$SOURCE"
else
  git -C "$SOURCE" pull --ff-only
fi

# uv installs its Python 3.11 interpreter into its own managed directory; this
# does not replace or modify the system Python, or ComfyUI's Python 3.12.
"$UV_BIN" python install 3.11

export HF_HOME="$MODELS_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODELSCOPE_CACHE="$MODELS_DIR/modelscope"

echo "Installing IndexTTS 2.5 dependencies in $SOURCE/.venv (Python 3.11)"
(
  cd "$SOURCE"
  "$UV_BIN" sync --python 3.11 --no-dev
)
"$SOURCE/.venv/bin/python" -c 'from indextts.infer_v2_5 import IndexTTS2; print("IndexTTS 2.5 backend import succeeded")'

echo
echo "IndexTTS 2.5 installed in its own Python 3.11 environment."
echo "Download checkpoint weights using TTS API -> Server -> IndexTTS 2.5 -> Download weights."
echo "Weights: $INDEX_ROOT/checkpoints"
echo "Auxiliary model caches will also be stored under checkpoints/hf_cache."
echo "Restart ComfyUI after pulling the updated custom node."
