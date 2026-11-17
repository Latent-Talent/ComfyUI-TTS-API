#!/usr/bin/env bash
# Optional NON-DESTRUCTIVE migration of already downloaded Hugging Face TTS
# weights into ComfyUI/models/TTS. Existing shared caches remain untouched.
set -euo pipefail
cd "$(dirname "$0")"
COMFY_ROOT="$(cd ../.. && pwd)"
MODELS_DIR="${TTS_MODELS_DIR:-$COMFY_ROOT/models/TTS}"
TARGET="$MODELS_DIR/huggingface/hub"
SOURCE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
mkdir -p "$TARGET"
SOURCE="$(realpath -m "$SOURCE")"
TARGET="$(realpath -m "$TARGET")"
if [[ "$SOURCE" == "$TARGET" ]]; then
  echo "TTS already uses this Hugging Face cache: $TARGET"
  exit 0
fi
if [[ ! -d "$SOURCE" ]]; then
  echo "No legacy Hugging Face cache detected: $SOURCE"
  exit 0
fi
echo "Migrating existing TTS weights from:"
echo "  $SOURCE"
echo "into:"
echo "  $TARGET"
echo "This script never removes or overwrites your old cache."
count=0
for repo in \
    'models--openbmb--VoxCPM2' \
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice' \
    'models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice' \
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign' \
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-Base' \
    'models--Qwen--Qwen3-TTS-12Hz-0.6B-Base'
do
  [[ -d "$SOURCE/$repo" ]] || continue
  if [[ -e "$TARGET/$repo" ]]; then
    echo "Keeping existing destination: $repo"
    continue
  fi
  temp="$TARGET/.migrating-${repo}-$$"
  if [[ "$(stat -c '%d' "$SOURCE")" == "$(stat -c '%d' "$TARGET")" ]]; then
    # The HF files are immutable blobs. Hard links reuse the bytes, preserve
    # snapshot symlinks, and avoid consuming a second copy on the same disk.
    cp -al "$SOURCE/$repo" "$temp"
  else
    cp -a "$SOURCE/$repo" "$temp"
  fi
  mv "$temp" "$TARGET/$repo"
  echo "Reused checkpoint cache: $repo"
  count=$((count + 1))
done
echo "Migration completed: $count TTS checkpoint caches staged in $TARGET."
echo "Original shared Hugging Face cache is unchanged."
