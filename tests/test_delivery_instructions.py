"""Delivery instruction tests: only supplied text reaches final synthesis."""
import numpy as np
from fastapi.testclient import TestClient

from tts_api import server
from tts_api.engine import VoxEngine


class CapturingVox:
    def __init__(self):
        self.tts_model = type("AudioConfig", (), {"sample_rate": 48000})()
        self.calls = []

    def _generate(self, text, cfg_value=2.0, inference_timesteps=10,
                  reference_wav_path=None, prompt_text=None,
                  prompt_wav_path=None):
        self.calls.append({
            "text": text,
            "reference_wav_path": reference_wav_path,
            "prompt_text": prompt_text,
            "prompt_wav_path": prompt_wav_path,
        })
        return np.zeros(4800, dtype=np.float32)

    def generate(self, **kwargs):
        return self._generate(**kwargs)


def test_delivery_uses_disposable_reference_and_not_spoken_prompt(tmp_path):
    engine = VoxEngine("fake-model")
    model = CapturingVox()
    engine._model = model

    response = engine.generate_with_delivery(
        "This is the text I want to hear.",
        instructions="Warm and cheerful with a slightly faster pace",
    )
    assert response[:4] == b"RIFF"
    assert len(model.calls) == 2
    assert model.calls[0]["text"].startswith(
        "(Warm and cheerful with a slightly faster pace)"
    )
    assert model.calls[1]["text"] == "This is the text I want to hear."
    assert model.calls[1]["reference_wav_path"] is not None
    from pathlib import Path
    assert not Path(model.calls[1]["reference_wav_path"]).exists()

    engine.generate_with_delivery("A second reply.", instructions="Warm and cheerful with a slightly faster pace")
    # Scratch reference reused: only final synthesis happens on second call.
    assert len(model.calls) == 3
    assert model.calls[2]["text"] == "A second reply."
    assert len(engine._style_cache) == 1


def test_regular_speech_does_not_insert_description():
    engine = VoxEngine("fake-model")
    model = CapturingVox()
    engine._model = model
    engine.generate_with_delivery("Only this sentence.", instructions="")
    assert [item["text"] for item in model.calls] == ["Only this sentence."]


def test_delivery_api_returns_wav_and_only_sample_is_final(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    engine = VoxEngine("fake-model")
    model = CapturingVox()
    engine._model = model
    monkeypatch.setattr(server, "ENGINE", engine)
    client = TestClient(server.app)
    reply = client.post("/v1/audio/speech", json={
        "model": "voxcpm2",
        "input": "Read only these words.",
        "instructions": "Happy and upbeat",
        "response_format": "wav",
    })
    assert reply.status_code == 200, reply.text
    assert reply.content[:4] == b"RIFF"
    assert model.calls[-1]["text"] == "Read only these words."
    assert len(model.calls) == 2


def test_saved_voice_delivery_uses_reference_then_clean_text(monkeypatch, tmp_path):
    import io
    import soundfile as sf
    import numpy as np

    store = server.VoiceStore(tmp_path)
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(4800, dtype=np.float32), 48000, format="WAV")
    voice = store.create(
        name="Narrator", description="Deep voice",
        reference=buffer.getvalue(), reference_text="Hello, friend.",
    )
    monkeypatch.setattr(server, "STORE", store)
    engine = VoxEngine("fake-model")
    model = CapturingVox()
    engine._model = model
    monkeypatch.setattr(server, "ENGINE", engine)
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "voxcpm2",
        "input": "This exact sentence must be spoken.",
        "voice": voice["id"],
        "instructions": "A gentle, reassuring tone",
        "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert model.calls[0]["text"] == "(A gentle, reassuring tone)Hello there."
    assert model.calls[0]["reference_wav_path"] == str(store.audio_path(voice))
    assert model.calls[-1]["text"] == "This exact sentence must be spoken."
    assert model.calls[-1]["prompt_text"] is None
    assert model.calls[-1]["reference_wav_path"] != str(store.audio_path(voice))
