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

if ! command -v uv >/dev/null 2>&1; then
  echo "IndexTTS requires uv. Install uv first using https://docs.astral.sh/uv/getting-started/installation/." >&2
  echo "This installer will not change your existing Python installation." >&2
  exit 1
fi
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
uv python install 3.11

export HF_HOME="$MODELS_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODELSCOPE_CACHE="$MODELS_DIR/modelscope"

echo "Installing IndexTTS 2.5 dependencies in $SOURCE/.venv (Python 3.11)"
(
  cd "$SOURCE"
  uv sync --python 3.11 --no-dev
)
"$SOURCE/.venv/bin/python" -c 'from indextts.infer_v2_5 import IndexTTS2; print("IndexTTS 2.5 backend import succeeded")'

echo
echo "IndexTTS 2.5 installed in its own Python 3.11 environment."
echo "Download checkpoint weights using TTS API -> Server -> IndexTTS 2.5 -> Download weights."
echo "Weights: $INDEX_ROOT/checkpoints"
echo "Auxiliary model caches will also be stored under checkpoints/hf_cache."
echo "Restart ComfyUI after pulling the updated custom node."
