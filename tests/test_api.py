"""HTTP contract tests using a fake inference engine (no GPU/model weights)."""
import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from tts_api import server


def fake_wav():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(24000, dtype=np.float32), 48000, format="WAV")
    return buf.getvalue()


def test_health(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    client = TestClient(server.app)
    assert client.get("/health").json()["ok"] is True
    assert client.get("/v1/models").json()["data"][0]["id"] == "voxcpm2"


def test_speech_wav(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    monkeypatch.setattr(server, "_render", lambda req: fake_wav())
    client = TestClient(server.app)
    resp = client.post("/v1/audio/speech", json={
        "model": "voxcpm2", "input": "Testing", "response_format": "wav"
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content[:4] == b"RIFF"


def test_voice_clone_then_speech(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    monkeypatch.setattr(server, "_render", lambda req: fake_wav())
    client = TestClient(server.app)
    response = client.post(
        "/api/voices/clone",
        data={"name": "Test", "reference_text": "Hi", "consent": "true"},
        files={"audio": ("sample.wav", fake_wav(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    voice = response.json()
    assert voice["id"].startswith("voice_")
    listed = client.get("/v1/audio/voices").json()["voices"]
    assert any(v["id"] == voice["id"] for v in listed)
    assert client.delete(f"/api/voices/{voice['id']}").status_code == 200


def test_clone_requires_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    client = TestClient(server.app)
    resp = client.post("/api/voices/clone",
                       data={"name": "Test", "consent": "false"},
                       files={"audio": ("sample.wav", fake_wav(), "audio/wav")})
    assert resp.status_code == 422


def test_missing_voice_and_unsupported_model(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    client = TestClient(server.app)
    assert client.post("/v1/audio/speech", json={
        "model": "voxcpm2", "input": "Testing", "voice": "missing",
        "response_format": "wav"
    }).status_code == 404
    assert client.post("/v1/audio/speech", json={
        "model": "unavailable", "input": "Testing",
        "response_format": "wav"
    }).status_code == 422
