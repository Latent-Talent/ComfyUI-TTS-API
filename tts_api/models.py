"""Installed TTS model catalog and OpenAI-compatible model routing.

Pure metadata: safe to import by ComfyUI without torch / qwen dependencies.
"""
from __future__ import annotations

VOX_ID = "voxcpm2"
QWEN_DEFAULT = "qwen3-tts-1.7b-customvoice"
QWEN_DESIGN = "qwen3-tts-1.7b-voicedesign"
INDEX_ID = "indextts-2.5"
INDEX_REPO = "IndexTeam/IndexTTS-2.5"
INDEX_EMOTIONS = (
    "happy", "angry", "sad", "afraid", "disgusted",
    "melancholic", "surprised", "calm",
)

QWEN_MODELS = {
    "qwen3-tts-1.7b-customvoice": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "kind": "customvoice", "size": "1.7B", "label": "Qwen3 TTS · 1.7B CustomVoice",
    },
    "qwen3-tts-0.6b-customvoice": {
        "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "kind": "customvoice", "size": "0.6B", "label": "Qwen3 TTS · 0.6B CustomVoice",
    },
    "qwen3-tts-1.7b-voicedesign": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "kind": "voicedesign", "size": "1.7B", "label": "Qwen3 TTS · 1.7B VoiceDesign",
    },
    "qwen3-tts-1.7b-base": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "kind": "base", "size": "1.7B", "label": "Qwen3 TTS · 1.7B Base (cloning)",
    },
    "qwen3-tts-0.6b-base": {
        "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "kind": "base", "size": "0.6B", "label": "Qwen3 TTS · 0.6B Base (cloning)",
    },
}
QWEN_SPEAKERS = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)
QWEN_SPEAKERS_LOOKUP = {name.casefold(): name for name in QWEN_SPEAKERS}

OPENAI_MODEL_ALIASES = frozenset({
    "tts-1", "tts-1-hd", "tts-1-1106", "tts-1-hd-1106",
    "gpt-4o-mini-tts",
})
OPENAI_VOICE_ALIASES = frozenset({
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "ash", "ballad", "coral", "sage", "verse", "marin", "cedar",
})


def canonical_model(requested: str | None, default: str = VOX_ID,
                    voxcpm_checkpoint: str = "openbmb/VoxCPM2") -> str:
    """Resolve advertised model name to a known installed backend family."""
    candidate = (requested or "").strip()
    if candidate == voxcpm_checkpoint or candidate == VOX_ID:
        return VOX_ID
    if candidate in QWEN_MODELS:
        return candidate
    if candidate in (INDEX_ID, "indextts", "indextts2.5"):
        return INDEX_ID
    if candidate == "qwen3-tts":
        return QWEN_DEFAULT
    if candidate in OPENAI_MODEL_ALIASES or not candidate:
        return default
    raise ValueError(
        f"Unknown model {candidate!r}. Choose voxcpm2, qwen3-tts, indextts-2.5, "
        "or one of the IDs returned by /v1/models."
    )


def is_index(model_id: str) -> bool:
    return model_id == INDEX_ID


def is_qwen(model_id: str) -> bool:
    return model_id in QWEN_MODELS


def qwen_base_for(model_id: str) -> str:
    """Use 0.6B Base when 0.6B selected, otherwise 1.7B Base."""
    if model_id.startswith("qwen3-tts-0.6b"):
        return "qwen3-tts-0.6b-base"
    return "qwen3-tts-1.7b-base"


def model_catalog() -> list[dict]:
    return [
        {"id": VOX_ID, "label": "VoxCPM2", "kind": "voxcpm",
         "features": ["speech", "voice_design", "clone"]},
        {"id": INDEX_ID, "label": "IndexTTS · 2.5 (emotion + cloning)",
         "kind": "indextts", "size": "2.5", "repo": INDEX_REPO,
         "features": ["speech", "clone", "emotion_vectors", "emotion_from_text"]},
        *[
            {"id": name, "label": info["label"], "kind": info["kind"],
             "size": info["size"], "repo": info["repo"]}
            for name, info in QWEN_MODELS.items()
        ],
    ]
