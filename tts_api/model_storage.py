"""Centralized TTS checkpoint paths, independent of voice recordings.

Hugging Face, IndexTTS, and auxiliary model files use ComfyUI/models/TTS.
This module only resolves paths and configures the dedicated TTS worker cache.
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


def has_weight_files(checkpoint_dir: Path) -> bool:
    """Check local checkpoint contents without accessing the network.

    A config or an incomplete sharded download does not count as installed.
    """
    import json

    folder = Path(checkpoint_dir)
    indexes = [
        *folder.glob("*.safetensors.index.json"),
        *folder.glob("*.bin.index.json"),
    ]
    if indexes:
        for index_file in indexes:
            try:
                weights = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
                shards = set(weights.values())
                if not shards or not all(
                    (folder / shard).is_file() and (folder / shard).stat().st_size > 0
                    for shard in shards
                ):
                    return False
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return False
        return True

    return any(
        file.is_file() and file.stat().st_size > 0
        for suffix in ("*.safetensors", "*.bin", "*.pth", "*.pt")
        for file in folder.glob(suffix)
    )


def index_weights_downloaded() -> bool:
    """IndexTTS requires its config and actual GPT checkpoint, not just config."""
    import re

    folder = index_checkpoints()
    config = folder / "config.yaml"
    if not config.is_file():
        return False
    try:
        contents = config.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*gpt_checkpoint:\s*['\"]?([^'\\"\s#]+)", contents)
        if not match:
            return False
        checkpoint = (folder / match.group(1)).resolve()
        if not checkpoint.is_relative_to(folder.resolve()):
            return False
        return checkpoint.is_file() and checkpoint.stat().st_size > 0
    except (OSError, ValueError):
        return False


def qwen_weights_downloaded(repo: str) -> bool:
    """Look only in this worker's model cache; never trigger a download."""
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(
            repo, "config.json", cache_dir=str(hf_hub_cache())
        )
        if not isinstance(config, str) or not Path(config).is_file():
            return False
        return has_weight_files(Path(config).parent)
    except (ImportError, OSError, ValueError):
        return False
