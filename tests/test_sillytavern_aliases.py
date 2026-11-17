"""SillyTavern OpenAI-compatible model/voice defaults route to VoxCPM2."""
import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from tts_api import server


def wav_response():
    out = io.BytesIO()
    sf.write(out, np.zeros(4800, dtype=np.float32), 48000, format="WAV")
    return out.getvalue()


@pytest.mark.parametrize("model", [
    "voxcpm2", "tts-1", "tts-1-hd", "tts-1-1106",
    "tts-1-hd-1106", "gpt-4o-mini-tts",
])
@pytest.mark.parametrize("voice", ["default", "alloy", "echo", "nova", "shimmer"])
def test_sillytavern_aliases_reach_local_voxcpm2(monkeypatch, tmp_path, model, voice):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    calls = []

    def fake_inference(*args, **kwargs):
        calls.append((args, kwargs))
        return wav_response()

    monkeypatch.setattr(server.ENGINE, "generate_with_delivery", fake_inference)
    client = TestClient(server.app)
    response = client.post("/v1/audio/speech", json={
        "model": model,
        "voice": voice,
        "input": "Only this text.",
        "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert response.content[:4] == b"RIFF"
    assert len(calls) == 1
    assert calls[0][0][0] == "Only this text."
    assert calls[0][1]["reference_path"] is None


def test_list_models_exposes_aliases():
    response = TestClient(server.app).get("/v1/models")
    assert response.status_code == 200
    models = {entry["id"]: entry for entry in response.json()["data"]}
    assert models["voxcpm2"]["owned_by"] == "local"
    assert models["tts-1"]["owned_by"] == "local-active-backend-alias"
    assert models["gpt-4o-mini-tts"]["owned_by"] == "local-voxcpm2-alias"


def test_unknown_model_still_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "unknown-not-installed",
        "voice": "alloy",
        "input": "Hello",
        "response_format": "wav",
    })
    assert response.status_code == 422
    assert "voxcpm2" in response.json()["detail"]


def test_unknown_voice_still_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "tts-1",
        "voice": "voice_not_found",
        "input": "Hello",
        "response_format": "wav",
    })
    assert response.status_code == 404


def test_saved_voice_id_unaffected(monkeypatch, tmp_path):
    store = server.VoiceStore(tmp_path)
    voice = store.create("SillyTavern character")
    monkeypatch.setattr(server, "STORE", store)
    args = []
    def fake_inference(*a, **kw):
        args.append((a, kw))
        return wav_response()
    monkeypatch.setattr(server.ENGINE, "generate_with_delivery", fake_inference)
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "tts-1",
        "voice": voice["id"],
        "input": "Hello character",
        "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert args and args[0][0][0] == "Hello character"
