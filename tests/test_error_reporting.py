"""Regression tests for opaque HTTP 500s and VoxCPM graph compile fallback."""
import io
import os
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from tts_api import server
from tts_api.engine import VoxEngine


def test_unexpected_backend_exception_returns_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))

    def unexpected_error(_request):
        raise AssertionError("example tensor-size mismatch from backend")

    monkeypatch.setattr(server, "_render", unexpected_error)
    client = TestClient(server.app)
    result = client.post("/v1/audio/speech", json={
        "model": "voxcpm2",
        "input": "Hello",
        "response_format": "wav",
    })
    assert result.status_code == 503
    detail = result.json()["detail"]
    assert detail["exception"] == "AssertionError"
    assert "tensor-size mismatch" in detail["message"]
    assert detail["stage"] == "speech_api"


def test_fallback_only_on_known_compiled_cuda_errors(monkeypatch):
    engine = VoxEngine("fake-model")
    engine.optimize = True
    loaded = []

    class FakeModel:
        def __init__(self, optimized):
            self.optimized = optimized
            self.tts_model = type("SampleRate", (), {"sample_rate": 48000})()

        def generate(self, **kwargs):
            if self.optimized:
                raise RuntimeError(
                    "cudaMallocAsync does not yet support checkPoolLiveAllocations"
                )
            return np.zeros(4800, dtype=np.float32)

    def fake_load():
        loaded.append(engine.optimize)
        engine._model = FakeModel(engine.optimize)

    def fake_unload():
        engine._model = None

    monkeypatch.setattr(engine, "_load", fake_load)
    monkeypatch.setattr(engine, "_unload_locked", fake_unload)
    output = engine.generate("Hello", seed=42)
    assert output[:4] == b"RIFF"
    assert loaded == [True, False]
    assert engine.optimize is False
    assert engine.compile_fallback_used is True
    assert engine.last_error is None
    assert engine.last_stats["audio_seconds"] > 0


def test_unrelated_error_does_not_retry(monkeypatch):
    engine = VoxEngine("fake-model")
    engine.optimize = True
    calls = []

    class BadModel:
        def generate(self, **kwargs):
            calls.append(1)
            raise ValueError("unrelated invalid input")

    def fake_load():
        engine._model = BadModel()

    monkeypatch.setattr(engine, "_load", fake_load)
    try:
        engine.generate("Hello")
    except ValueError:
        pass
    else:
        raise AssertionError("Backend failure was swallowed")
    assert len(calls) == 1
    assert engine.optimize is True
    assert engine.last_error["message"] == "unrelated invalid input"


def test_status_exposes_last_error_without_loading_model(monkeypatch):
    engine = VoxEngine("fake-model")
    engine.last_error = {
        "stage": "generation",
        "type": "RuntimeError",
        "message": "failed kernel",
    }
    monkeypatch.setattr(server, "ENGINE", engine)
    client = TestClient(server.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["last_error"]["message"] == "failed kernel"
    assert response.json()["compile_enabled"] is True
