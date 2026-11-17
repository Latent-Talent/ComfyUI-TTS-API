"""Validated, persistent per-backend speech defaults for ComfyUI-TTS-API.

Profiles are independent for every advertised model ID. They only supply fields
omitted from speech API requests; explicitly supplied values always win.
Stored locally with mode 0600 and never forwarded to ComfyUI's job queue.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import threading

from .models import INDEX_EMOTIONS, INDEX_ID, QWEN_MODELS, VOX_ID

_LOCK = threading.RLock()
FORMATS = ("mp3", "wav", "opus", "aac", "flac", "pcm")
QWEN_LANGUAGES = (
    "Auto", "English", "Chinese", "Japanese", "Korean", "German", "French",
    "Spanish", "Arabic", "Italian", "Portuguese", "Russian",
)
INDEX_LANGUAGES = ("Auto", "English", "Chinese", "Japanese", "Spanish", "Arabic")


def field(name: str, label: str, kind: str, default, **options) -> dict:
    return {"name": name, "label": label, "kind": kind,
            "default": default, **options}


def common_fields(*, include_voice: bool = True) -> list[dict]:
    fields = [
        field("response_format", "Audio format", "select", "mp3", options=list(FORMATS)),
        field("speed", "Playback speed", "number", 1.0, min=0.25, max=4.0, step=0.05),
        field("seed", "Seed (blank = random)", "nullable_integer", None,
              min=0, max=2147483647),
    ]
    if include_voice:
        fields.insert(0, field(
            "voice", "Default voice", "audio_voice", "default", max_length=120,
            hint="Select a saved voice or browse for a reference recording.",
            audio_reference=True,
        ))
    return fields




def schema_for(model_id: str) -> list[dict]:
    if model_id == VOX_ID:
        return common_fields() + [
            field("instructions", "Default delivery instructions", "textarea", "",
                  max_length=1000, hint="Optional style instruction"),
            field("steps", "Inference steps", "integer", 10, min=1, max=100),
            field("cfg", "CFG strength", "number", 2.0, min=0, max=10, step=0.1),
        ]
    if model_id == INDEX_ID:
        return common_fields() + [
            field("language", "Language", "select", "Auto",
                  options=list(INDEX_LANGUAGES)),
            field("emotion_mode", "Emotion source", "select", "neutral",
                  options=["neutral", "auto", "text", "vector", "reference"]),
            field("instructions", "Emotion description", "textarea", "",
                  max_length=1000, hint="Used with emotion source: text",
                  visible_when={"emotion_mode": ["text"]}),
            field("emotion_vector", "Emotion vector", "vector",
                  [0.0] * len(INDEX_EMOTIONS),
                  emotions=list(INDEX_EMOTIONS), min=0.0, max=1.0, step=0.05,
                  visible_when={"emotion_mode": ["vector"]}),
            field("emotion_strength", "Emotion strength", "number", 0.65,
                  min=0.0, max=1.0, step=0.05,
                  visible_when={"emotion_mode": ["auto", "text", "vector", "reference"]}),
            field("emotion_reference_voice", "Emotion reference audio", "audio_voice", "",
                  max_length=120, hint="Choose a saved recording or browse for an audio file.",
                  audio_reference=True, required_saved_voice=True,
                  visible_when={"emotion_mode": ["reference"]}),
            field("duration_factor", "Duration factor", "number", 1.0,
                  min=0.5, max=2.0, step=0.05),
        ]
    if model_id in QWEN_MODELS:
        kind = QWEN_MODELS[model_id]["kind"]
        result = common_fields(include_voice=kind != "voicedesign") + [
            field("language", "Language", "select", "Auto",
                  options=list(QWEN_LANGUAGES)),
        ]
        if kind == "voicedesign" or (kind == "customvoice" and "1.7b" in model_id):
            result.append(
                field("instructions", "Default voice/delivery instructions",
                      "textarea", "", max_length=1000,
                      hint="VoiceDesign uses this as a voice description"),
            )
        return result
    raise ValueError("Unknown TTS model ID")


def builtins_for(model_id: str) -> dict:
    return {item["name"]: copy.deepcopy(item["default"])
            for item in schema_for(model_id)}


def _validate_value(spec: dict, value):
    kind = spec["kind"]
    if kind in ("text", "textarea", "audio_voice"):
        if type(value) is not str or len(value) > spec.get("max_length", 1000):
            raise ValueError("Invalid model default: " + spec["name"])
        return value
    if kind == "select":
        if type(value) is not str or value not in spec["options"]:
            raise ValueError("Invalid model default: " + spec["name"])
        return value
    if kind == "vector":
        emotions = spec["emotions"]
        if (not isinstance(value, list) or len(value) != len(emotions) or
                any(type(number) not in (int, float) or
                    not math.isfinite(number) or
                    not spec["min"] <= number <= spec["max"] for number in value)):
            raise ValueError("Invalid model default: " + spec["name"])
        return [float(number) for number in value]
    if kind == "nullable_integer" and value is None:
        return None
    if kind in ("integer", "nullable_integer"):
        if type(value) is not int or not spec["min"] <= value <= spec["max"]:
            raise ValueError("Invalid model default: " + spec["name"])
        return value
    if kind == "number":
        if (type(value) not in (int, float) or not math.isfinite(value) or
                not spec["min"] <= value <= spec["max"]):
            raise ValueError("Invalid model default: " + spec["name"])
        return float(value)
    raise ValueError("Unsupported model default type")


def validate_settings(model_id: str, settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Model defaults must be a JSON object")
    fields = {spec["name"]: spec for spec in schema_for(model_id)}
    unknown = set(settings) - set(fields)
    if unknown:
        raise ValueError("Unsupported default for selected TTS model")
    return {key: _validate_value(fields[key], value)
            for key, value in settings.items()}


def _file_path(data_root: Path) -> Path:
    return Path(data_root) / "model_defaults.json"


def _load_file(data_root: Path) -> dict:
    path = _file_path(data_root)
    try:
        if not path.is_file() or path.stat().st_size > 131072:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def effective_defaults(data_root: Path, model_id: str) -> dict:
    values = builtins_for(model_id)
    with _LOCK:
        saved = _load_file(data_root).get(model_id, {})
    if isinstance(saved, dict):
        # Revalidate manually edited files so malformed settings cannot crash
        # speech requests or bypass the backend's supported field definitions.
        for name, value in saved.items():
            try:
                values.update(validate_settings(model_id, {name: value}))
            except ValueError:
                continue
    return values


def save_defaults(data_root: Path, model_id: str,
                  settings: dict, *, reset: bool = False) -> dict:
    schema_for(model_id)
    valid = validate_settings(model_id, settings)
    data_root = Path(data_root)
    with _LOCK:
        data_root.mkdir(parents=True, exist_ok=True)
        store = _load_file(data_root)
        if reset:
            store.pop(model_id, None)
        else:
            prior = store.get(model_id, {})
            prior = prior if isinstance(prior, dict) else {}
            merged = builtins_for(model_id)
            for key, val in prior.items():
                try:
                    merged.update(validate_settings(model_id, {key: val}))
                except ValueError:
                    continue
            merged.update(valid)
            store[model_id] = merged
        fd, temp_path = tempfile.mkstemp(prefix=".model-defaults-", dir=data_root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(store, stream, ensure_ascii=False,
                          indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, _file_path(data_root))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return effective_defaults(data_root, model_id)


def all_defaults(data_root: Path) -> dict:
    return {
        model_id: {"fields": schema_for(model_id),
                   "values": effective_defaults(data_root, model_id)}
        for model_id in (VOX_ID, INDEX_ID, *QWEN_MODELS)
    }
