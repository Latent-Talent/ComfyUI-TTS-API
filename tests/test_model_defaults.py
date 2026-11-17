"""Model-specific defaults, override precedence, and downloaded-model detection."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tts_api import model_storage, server
from tts_api.model_defaults import (
    all_defaults, builtins_for, effective_defaults, save_defaults,
    schema_for, validate_settings,
)
from tts_api.models import INDEX_ID, QWEN_MODELS, VOX_ID


def test_schema_is_unique_and_backend_specific_for_each_model():
    profiles = all_defaults(Path("/unused-model-default-directory"))
    assert set(profiles) == {VOX_ID, INDEX_ID, *QWEN_MODELS}
    vox = builtins_for(VOX_ID)
    index = builtins_for(INDEX_ID)
    custom_17 = builtins_for("qwen3-tts-1.7b-customvoice")
    custom_06 = builtins_for("qwen3-tts-0.6b-customvoice")
    qwen_base = builtins_for("qwen3-tts-1.7b-base")
    assert {"steps", "cfg", "instructions"} <= set(vox)
    assert "steps" not in custom_17
    assert "instructions" in custom_17
    assert "instructions" not in custom_06
    assert "instructions" not in qwen_base
    assert "emotion_vector" in index and len(index["emotion_vector"]) == 8
    assert "emotion_mode" not in vox
    assert index["emotion_mode"] == "neutral"
    assert index["duration_factor"] == 1.0


def test_save_load_reset_per_model_and_private_file(tmp_path):
    assert effective_defaults(tmp_path, VOX_ID)["steps"] == 10
    changed = save_defaults(tmp_path, VOX_ID, {
        "steps": 22, "cfg": 3.5, "seed": 24, "instructions": "Soft delivery",
    })
    assert changed["steps"] == 22
    assert changed["cfg"] == 3.5
    assert changed["seed"] == 24
    saved = tmp_path / "model_defaults.json"
    assert saved.is_file()
    if os.name == "posix":
        assert saved.stat().st_mode & 0o777 == 0o600
    save_defaults(tmp_path, INDEX_ID, {
        "emotion_mode": "vector",
        "emotion_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    })
    assert effective_defaults(tmp_path, INDEX_ID)["emotion_mode"] == "vector"
    assert effective_defaults(tmp_path, VOX_ID)["steps"] == 22
    save_defaults(tmp_path, VOX_ID, {"voice": "alloy"})
    assert effective_defaults(tmp_path, VOX_ID)["steps"] == 22
    assert effective_defaults(tmp_path, VOX_ID)["voice"] == "alloy"
    assert save_defaults(tmp_path, VOX_ID, {}, reset=True) == builtins_for(VOX_ID)
    assert effective_defaults(tmp_path, INDEX_ID)["emotion_mode"] == "vector"


@pytest.mark.parametrize("model,settings", [
    (VOX_ID, {"steps": 0}),
    (VOX_ID, {"steps": 4.5}),
    (VOX_ID, {"cfg": 99}),
    (VOX_ID, {"seed": -1}),
    (VOX_ID, {"speed": float("nan")}),
    (VOX_ID, {"speed": float("inf")}),
    (VOX_ID, {"unexpected": 42}),
    (VOX_ID, {"response_format": "exe"}),
    (INDEX_ID, {"emotion_vector": [0] * 7}),
    (INDEX_ID, {"emotion_vector": [0] * 7 + [1.2]}),
    (INDEX_ID, {"emotion_mode": "unsafe"}),
    ("qwen3-tts-0.6b-customvoice", {"instructions": "Not supported"}),
])
def test_invalid_settings_rejected(model, settings):
    with pytest.raises(ValueError):
        validate_settings(model, settings)


def test_effective_defaults_ignore_corrupt_manual_edits(tmp_path):
    path = tmp_path / "model_defaults.json"
    path.write_text('{"voxcpm2": {"steps": 15, "cfg": 100, "unknown": 3}}')
    values = effective_defaults(tmp_path, VOX_ID)
    assert values["steps"] == 15
    assert values["cfg"] == 2.0
    assert "unknown" not in values
    path.write_text("not valid json")
    assert effective_defaults(tmp_path, VOX_ID) == builtins_for(VOX_ID)


def test_speech_request_uses_saved_values_only_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    save_defaults(tmp_path, VOX_ID, {
        "seed": 31, "steps": 28, "cfg": 4.2, "voice": "alloy",
        "response_format": "wav", "instructions": "gentle",
    })
    missing = server.apply_speech_defaults(
        server.SpeechRequest(model=VOX_ID, input="Testing")
    )
    assert missing.seed == 31
    assert missing.steps == 28
    assert missing.cfg == 4.2
    assert missing.response_format == "wav"
    assert missing.voice == "alloy"
    assert missing.instructions == "gentle"

    explicit = server.apply_speech_defaults(server.SpeechRequest(
        model=VOX_ID, input="Testing", steps=8, cfg=1.1, seed=None,
        response_format="mp3", voice="nova", instructions="",
    ))
    assert explicit.steps == 8
    assert explicit.cfg == 1.1
    assert explicit.seed is None
    assert explicit.response_format == "mp3"
    assert explicit.voice == "nova"
    assert explicit.instructions == ""

    save_defaults(tmp_path, INDEX_ID, {
        "emotion_mode": "auto", "emotion_strength": 0.9,
        "language": "Chinese", "duration_factor": 1.2,
    })
    index = server.apply_speech_defaults(
        server.SpeechRequest(model=INDEX_ID, input="Testing")
    )
    assert (index.emotion_mode, index.emotion_strength,
            index.language, index.duration_factor) == ("auto", 0.9, "Chinese", 1.2)
    assert server.apply_speech_defaults(server.SpeechRequest(
        model=INDEX_ID, input="Testing", emotion_mode="neutral",
    )).emotion_mode == "neutral"


def test_authenticated_model_defaults_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "KEY", "")
    client = TestClient(server.app)
    models = client.get("/api/models/defaults")
    assert models.status_code == 200, models.text
    assert models.json()["models"][VOX_ID]["values"]["steps"] == 10
    updated = client.post("/api/models/defaults", json={
        "model_id": "qwen3-tts-1.7b-customvoice",
        "settings": {"voice": "Serena", "language": "Japanese"},
    })
    assert updated.status_code == 200, updated.text
    assert updated.json()["settings"]["voice"] == "Serena"
    assert client.get("/api/models/defaults").json()["models"][
        "qwen3-tts-1.7b-customvoice"]["values"]["language"] == "Japanese"
    bad = client.post("/api/models/defaults", json={
        "model_id": VOX_ID, "settings": {"steps": -3},
    })
    assert bad.status_code == 422
    assert client.post("/api/models/defaults", json={
        "model_id": VOX_ID, "settings": {}, "reset": True,
    }).status_code == 200


def test_index_checkpoint_detection_rejects_incomplete_download(tmp_path, monkeypatch):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    monkeypatch.setattr(model_storage, "index_checkpoints", lambda: folder)
    (folder / "config.yaml").write_text(
        "gpt_checkpoint: gpt.pth\ns2mel_checkpoint: s2mel.pth\n"
        "w2v_stat: w2v.pt\nqwen_emo_path: qwen-emo\n"
    )
    assert not model_storage.index_weights_downloaded()
    for name in ("gpt.pth", "s2mel.pth", "w2v.pt", "codec.pth"):
        (folder / name).write_bytes(b"weights")
    assert not model_storage.index_weights_downloaded()
    (folder / "qwen-emo").mkdir()
    assert model_storage.index_weights_downloaded()


def test_qwen_checkpoint_detection_rejects_missing_shards(tmp_path):
    checkpoint = tmp_path / "snapshots" / "revision"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    assert not model_storage.has_weight_files(checkpoint)
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"part1.safetensors","b":"part2.safetensors"}}'
    )
    (checkpoint / "part1.safetensors").write_bytes(b"weights")
    assert not model_storage.has_weight_files(checkpoint)
    (checkpoint / "part2.safetensors").write_bytes(b"weights")
    assert model_storage.has_weight_files(checkpoint)


def test_model_catalog_exposes_downloaded_flag_without_network(monkeypatch):
    monkeypatch.setattr(server, "KEY", "")
    monkeypatch.setattr(server, "index_weights_downloaded", lambda: True)
    monkeypatch.setattr(server, "qwen_weights_downloaded", lambda repo: "0.6B" in repo)
    records = TestClient(server.app).get("/api/models").json()["models"]
    indexed = {record["id"]: record["cache_detected"] for record in records}
    assert indexed[VOX_ID] is True
    assert indexed[INDEX_ID] is True
    assert indexed["qwen3-tts-0.6b-base"] is True
    assert indexed["qwen3-tts-1.7b-base"] is False


def test_indextts_dynamic_fields_are_mode_specific():
    fields = {spec["name"]: spec for spec in schema_for(INDEX_ID)}
    assert fields["voice"]["kind"] == "audio_voice"
    assert fields["voice"]["audio_reference"] is True
    assert fields["emotion_reference_voice"]["kind"] == "audio_voice"
    assert fields["emotion_reference_voice"]["visible_when"] == {
        "emotion_mode": ["reference"]
    }
    assert fields["emotion_vector"]["visible_when"] == {
        "emotion_mode": ["vector"]
    }
    assert fields["instructions"]["visible_when"] == {
        "emotion_mode": ["text"]
    }
    assert fields["emotion_strength"]["visible_when"] == {
        "emotion_mode": ["auto", "text", "vector", "reference"]
    }
    assert "visible_when" not in fields["duration_factor"]
    assert "voice" not in builtins_for("qwen3-tts-1.7b-voicedesign")
    assert fields["emotion_reference_voice"]["required_saved_voice"] is True


def test_audio_fields_store_voice_ids_not_file_contents(tmp_path):
    values = save_defaults(tmp_path, INDEX_ID, {
        "voice": "voice_0123456789abcdef",
        "emotion_reference_voice": "voice_abcdef0123456789",
        "emotion_mode": "reference",
    })
    assert values["voice"] == "voice_0123456789abcdef"
    assert values["emotion_reference_voice"] == "voice_abcdef0123456789"
    stored = (tmp_path / "model_defaults.json").read_text()
    assert "voice_0123456789abcdef" in stored
    assert '"emotion_mode": "reference"' in stored
    assert '"audio"' not in stored


def test_indextts_explicit_neutral_cannot_be_overridden_by_hidden_description(
        tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    save_defaults(tmp_path, INDEX_ID, {
        "emotion_mode": "neutral", "instructions": "An earlier angry style",
    })
    inferred = server.apply_speech_defaults(
        server.SpeechRequest(model=INDEX_ID, input="Sample")
    )
    assert inferred.emotion_mode == "neutral"
    assert inferred.instructions == "An earlier angry style"

    explicit = server.apply_speech_defaults(server.SpeechRequest(
        model=INDEX_ID, input="Sample", emotion_mode="neutral",
        instructions="This should not override neutral",
    ))
    assert explicit.emotion_mode == "neutral"
    legacy = server.apply_speech_defaults(server.SpeechRequest(
        model=INDEX_ID, input="Sample", instructions="Speak softly",
    ))
    assert legacy.emotion_mode == "text"
