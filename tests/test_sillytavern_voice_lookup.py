"""Voice lookup compatibility across SillyTavern and saved voice library."""
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tts_api import server


def _wav():
    buff = io.BytesIO()
    sf.write(buff, np.zeros(4800, dtype=np.float32), 48000, format="WAV")
    return buff.getvalue()


def _client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    generated = []
    def generate(*args, **kwargs):
        generated.append((args, kwargs))
        return _wav()
    monkeypatch.setattr(server.ENGINE, "generate_with_delivery", generate)
    return TestClient(server.app), generated


def _speak(client, voice):
    return client.post("/v1/audio/speech", json={
        "model": "tts-1", "voice": voice, "input": "Hello world.",
        "response_format": "wav",
    })


def test_builtin_aliases_trimmed_and_case_insensitive(monkeypatch, tmp_path):
    client, generated = _client(monkeypatch, tmp_path)
    for name in ["alloy", " Alloy ", " NOVA ", "Default ", "echo  ", "shimmer"]:
        result = _speak(client, name)
        assert result.status_code == 200, (name, result.text)
        assert result.content.startswith(b"RIFF")
    assert all(args[1]["reference_path"] is None for args in generated)


def test_saved_voice_matches_display_name_or_id(monkeypatch, tmp_path):
    client, generated = _client(monkeypatch, tmp_path)
    saved = server.STORE.create("My Character Voice", reference=_wav(),
                                reference_text="Hello, my friend.")
    for name in [saved["id"], "My Character Voice", " my character voice  "]:
        response = _speak(client, name)
        assert response.status_code == 200, response.text
        assert generated[-1][1]["reference_path"] == str(server.STORE.audio_path(saved))
        assert generated[-1][1]["reference_text"] == "Hello, my friend."


def test_unknown_voice_error_includes_requested_value(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    result = _speak(client, "SillyTavern Unmapped Voice")
    assert result.status_code == 404
    message = result.json()["detail"]
    assert "SillyTavern Unmapped Voice" in message
    assert "/v1/audio/voices" in message
    assert "default" in message


def test_ambiguous_saved_voice_name_requires_id(monkeypatch, tmp_path):
    client, generated = _client(monkeypatch, tmp_path)
    first = server.STORE.create("Same Name")
    second = server.STORE.create("same name")
    ambiguous = _speak(client, "SAME NAME")
    assert ambiguous.status_code == 422
    assert first["id"] in ambiguous.json()["detail"]
    assert second["id"] in ambiguous.json()["detail"]
    assert generated == []
    assert _speak(client, first["id"]).status_code == 200


def test_health_version(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == "0.4.1"
