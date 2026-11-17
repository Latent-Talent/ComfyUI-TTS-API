#!/usr/bin/env bash
# Move previously downloaded TTS weights. This is explicitly user-run;
# neither the API nor ComfyUI migrates old cache files automatically.
# Usage: bash move-tts-models.sh [--dry-run] [--index-source /old/checkpoints]
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash move-tts-models.sh [--dry-run] [--index-source /old/index/checkpoints]" \
    "Moves known TTS Hugging Face cache folders into ComfyUI/models/TTS." \
    "Use --index-source to move a separate existing IndexTTS checkpoints directory."
}
dry_run=0
index_source=""
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --index-source)
      (($# >= 2)) || { usage >&2; exit 2; }
      index_source=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

project_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
comfy_dir=$(cd -- "$project_dir/../.." && pwd -P)
models_dir=${TTS_MODELS_DIR:-"$comfy_dir/models/TTS"}
target=$(realpath -m -- "$models_dir/huggingface/hub")
source_cache=${HF_HUB_CACHE:-"${HF_HOME:-$HOME/.cache/huggingface}/hub"}
source=$(realpath -m -- "$source_cache")
index_target=$(realpath -m -- "$models_dir/IndexTTS-2.5/checkpoints")

if (( ! dry_run )) && pgrep -f '[u]vicorn tts_api.server:app' >/dev/null 2>&1; then
  printf '%s\n' \
    "TTS API appears to be running. Stop ComfyUI / the TTS server before moving model files." >&2
  exit 1
fi
printf 'Old Hugging Face cache: %s\n' "$source"
printf 'New TTS Hugging Face cache: %s\n' "$target"
printf 'IndexTTS checkpoints: %s\n' "$index_target"
if ((dry_run)); then printf '%s\n' 'DRY RUN: no files will change.'; fi

moved=0
skipped=0
if [[ "$source" == "$target" ]]; then
  printf '%s\n' "Source already equals destination; no Hugging Face cache move needed."
elif [[ -d "$source" ]]; then
  for repo in \
    'models--openbmb--VoxCPM2' \
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice' \
    'models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice' \
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign' \
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-Base' \
    'models--Qwen--Qwen3-TTS-12Hz-0.6B-Base' \
    'models--IndexTeam--IndexTTS-2.5'
  do
    from="$source/$repo"
    to="$target/$repo"
    if [[ ! -e "$from" && ! -L "$from" ]]; then continue; fi
    if [[ -e "$to" || -L "$to" ]]; then
      printf 'SKIP (destination exists): %s\n' "$repo"
      ((skipped+=1))
      continue
    fi
    printf 'MOVE: %s\n' "$repo"
    if (( ! dry_run )); then
      mkdir -p -- "$target"
      mv -T -- "$from" "$to"
    fi
    ((moved+=1))
  done
else
  printf '%s\n' "Old Hugging Face cache does not exist; nothing to move there."
fi

# A separate IndexTTS checkpoints directory is moved only when explicitly
# specified. The destination may exist due to install-index.sh.
if [[ -n "$index_source" ]]; then
  if [[ ! -d "$index_source" ]]; then
    printf 'IndexTTS source does not exist: %s\n' "$index_source" >&2
    exit 1
  fi
  index_source=$(realpath -- "$index_source")
  if [[ "$index_source" != "$index_target" ]]; then
    if [[ ! -f "$index_source/config.yaml" ]]; then
      printf 'Missing config.yaml in IndexTTS source: %s\n' "$index_source" >&2
      exit 1
    fi
    while IFS= read -r -d '' entry; do
      filename=${entry##*/}
      destination="$index_target/$filename"
      if [[ -e "$destination" || -L "$destination" ]]; then
        printf 'SKIP (IndexTTS destination exists): %s\n' "$filename"
        ((skipped+=1))
        continue
      fi
      printf 'MOVE IndexTTS: %s\n' "$filename"
      if (( ! dry_run )); then
        mkdir -p -- "$index_target"
        mv -- "$entry" "$destination"
      fi
      ((moved+=1))
    done < <(find "$index_source" -mindepth 1 -maxdepth 1 -print0)
  fi
fi

printf 'Finished: %d %s; %d skipped.\n' "$moved" "$([[ $dry_run -eq 1 ]] && echo planned || echo moved)" "$skipped"
printf '%s\n' 'Only TTS model directories were selected. Unrelated cached models were not touched.'
