"""Mocked Qwen3-TTS integration: no GPU or model download required."""
from __future__ import annotations

import io
from pathlib import Path
import types

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf
import pytest

from managed_service import ManagedTTS, read_config
from tts_api import server
from tts_api.models import (
    VOX_ID, QWEN_DEFAULT, QWEN_DESIGN, QWEN_MODELS,
    canonical_model, qwen_base_for, model_catalog,
)
from tts_api.qwen_engine import QwenEngine


def fake_wav() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(4800, dtype=np.float32), 48000, format="WAV")
    return buf.getvalue()


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(("custom", kwargs))
        return [np.zeros(4800, dtype=np.float32)], 48000

    def generate_voice_design(self, **kwargs):
        self.calls.append(("design", kwargs))
        return [np.zeros(4800, dtype=np.float32)], 48000

    def create_voice_clone_prompt(self, **kwargs):
        self.calls.append(("prompt", kwargs))
        return [{"ref_audio": kwargs["ref_audio"]}]

    def generate_voice_clone(self, **kwargs):
        self.calls.append(("clone", kwargs))
        return [np.zeros(4800, dtype=np.float32)], 48000


def fake_engine(monkeypatch):
    engine = QwenEngine()
    model = FakeModel()

    def load(model_id):
        engine._model = model
        engine._model_id = model_id

    monkeypatch.setattr(engine, "_load", load)
    return engine, model


@pytest.mark.parametrize("model", list(QWEN_MODELS))
def test_model_catalog_ids_are_distinct_and_canonical(model):
    assert canonical_model(model) == model
    assert QWEN_MODELS[model]["repo"].startswith("Qwen/Qwen3-TTS-12Hz-")
    assert any(item["id"] == model for item in model_catalog())


def test_default_and_tts_1_alias_can_route_to_qwen():
    assert canonical_model("tts-1", QWEN_DEFAULT) == QWEN_DEFAULT
    assert canonical_model("tts-1-hd", QWEN_DEFAULT) == QWEN_DEFAULT
    assert canonical_model("qwen3-tts", VOX_ID) == QWEN_DEFAULT


def test_builtin_speaker_uses_qwen_customvoice(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    engine, model = fake_engine(monkeypatch)
    monkeypatch.setattr(server, "QWEN_ENGINE", engine)
    request = {
        "model": QWEN_DEFAULT, "voice": "serena",
        "input": "Only the requested text.", "instructions": "Calm and steady",
        "response_format": "wav",
    }
    result = TestClient(server.app).post("/v1/audio/speech", json=request)
    assert result.status_code == 200, result.text
    assert result.content.startswith(b"RIFF")
    assert model.calls == [("custom", {
        "text": request["input"], "speaker": "Serena",
        "language": "Auto", "instruct": "Calm and steady",
    })]


def test_openai_alias_uses_configured_qwen_default(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    monkeypatch.setattr(server, "ACTIVE_MODEL", QWEN_DEFAULT)
    engine, model = fake_engine(monkeypatch)
    monkeypatch.setattr(server, "QWEN_ENGINE", engine)
    result = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "tts-1", "voice": "alloy",
        "input": "Local text", "response_format": "wav",
    })
    assert result.status_code == 200, result.text
    assert model.calls[0][1]["speaker"] == "Ryan"


def test_clone_reuses_saved_recording_and_qwen_prompt(monkeypatch, tmp_path):
    store = server.VoiceStore(tmp_path)
    record = store.create(name="Character", reference=fake_wav(), reference_text="Old words.")
    monkeypatch.setattr(server, "STORE", store)
    engine, model = fake_engine(monkeypatch)
    monkeypatch.setattr(server, "QWEN_ENGINE", engine)
    client = TestClient(server.app)
    for text in ["First sentence.", "Second sentence."]:
        result = client.post("/v1/audio/speech", json={
            "model": QWEN_DEFAULT, "voice": record["id"], "input": text,
            "response_format": "wav",
        })
        assert result.status_code == 200, result.text
    assert [kind for kind, _ in model.calls] == ["prompt", "clone", "clone"]
    assert model.calls[0][1]["ref_text"] == "Old words."
    assert model.calls[1][1]["text"] == "First sentence."
    assert model.calls[2][1]["text"] == "Second sentence."
    assert engine.loaded_model == qwen_base_for(QWEN_DEFAULT)


def test_qwen_design_only_synthesizes_sample_text(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    engine, model = fake_engine(monkeypatch)
    monkeypatch.setattr(server, "QWEN_ENGINE", engine)
    result = TestClient(server.app).post("/api/voices/design", json={
        "model": QWEN_DEFAULT, "name": "Made by Qwen",
        "description": "Mature warm British female voice",
        "sample_text": "These exact words.",
    })
    assert result.status_code == 200, result.text
    voice = result.json()
    assert server.STORE.audio_path(voice).is_file()
    assert model.calls == [("design", {
        "text": "These exact words.",
        "instruct": "Mature warm British female voice",
        "language": "Auto",
    })]
    assert engine.loaded_model == QWEN_DESIGN


def test_base_requires_saved_voice_not_default(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    engine, _ = fake_engine(monkeypatch)
    monkeypatch.setattr(server, "QWEN_ENGINE", engine)
    resp = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "qwen3-tts-0.6b-base",
        "voice": "default", "input": "Hello", "response_format": "wav",
    })
    assert resp.status_code == 422
    assert "requires a saved voice recording" in resp.json()["detail"]


def test_model_selection_preserves_api_key_and_emits_valid_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TTS_ACTIVE_MODEL", raising=False)
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "TTS_API_KEY=existing-long-secret-key\nTTS_ACTIVE_MODEL=voxcpm2\n",
        encoding="utf-8",
    )
    manager = ManagedTTS(root=tmp_path)
    manager._queue_busy = lambda: False
    manager._port_open = lambda: False
    manager._spawn = lambda: True
    response = manager.configure_active_model(QWEN_DEFAULT)
    assert response == {"active_model": QWEN_DEFAULT, "restarting": True}
    config = read_config(tmp_path)
    assert config["TTS_API_KEY"] == "existing-long-secret-key"
    assert config["TTS_ACTIVE_MODEL"] == QWEN_DEFAULT
    assert (tmp_path / ".env").read_text().count("TTS_ACTIVE_MODEL=") == 1
    assert manager.configure_active_model(QWEN_DEFAULT)["restarting"] is False


def test_reject_invalid_model_without_writing(monkeypatch, tmp_path):
    monkeypatch.delenv("TTS_ACTIVE_MODEL", raising=False)
    manager = ManagedTTS(root=tmp_path)
    with pytest.raises(ValueError, match="Unknown model"):
        manager.configure_active_model("../../malicious")
    assert not (tmp_path / ".env").exists()
