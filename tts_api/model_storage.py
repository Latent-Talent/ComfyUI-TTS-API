"""Centralized TTS checkpoint paths, independent of voice recordings.

All *new* model downloads, including Hugging Face hub and IndexTTS auxiliary
checkpoints, are rooted under ComfyUI/models/TTS. Existing legacy HF caches
remain untouched unless the user explicitly runs migrate-model-cache.sh.
"""
from __future__ import annotations

import os
from pathlib import Path


def model_root() -> Path:
    configured = os.environ.get("TTS_MODELS_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # .../ComfyUI/custom_nodes/ComfyUI-TTS-API/tts_api/model_storage.py
    return (Path(__file__).resolve().parents[3] / "models" / "TTS").resolve()


def hf_home() -> Path:
    return model_root() / "huggingface"


def hf_hub_cache() -> Path:
    return hf_home() / "hub"


def qwen_repo_path(repo: str) -> Path:
    return hf_hub_cache() / ("models--" + repo.replace("/", "--"))


def index_root() -> Path:
    return model_root() / "IndexTTS-2.5"


def index_checkpoints() -> Path:
    return index_root() / "checkpoints"


def index_source() -> Path:
    return index_root() / "source"


def index_python() -> Path:
    return index_source() / ".venv" / "bin" / "python"


def configure_model_storage() -> Path:
    """Configure this TTS worker's cache before importing Transformers/HF."""
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    # This is a dedicated process. Never write these variables to ComfyUI's
    # own os.environ or mutate its model/cache settings.
    os.environ["HF_HOME"] = str(hf_home())
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache())
    os.environ["MODELSCOPE_CACHE"] = str(model_root() / "modelscope")
    return root
