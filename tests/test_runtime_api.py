"""Verify the OpenAI speech API and Test overrides use saved per-model runtime settings.

The fake backend records the exact runtime supplied by _render; no GPU,
model download, quantization, or temporary installed Torch is required.
"""
from __future__ import annotations

import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tts_api import server
from tts_api.model_defaults import effective_defaults, save_defaults
from tts_api.models import VOX_ID, QWEN_DEFAULT
from tts_api.runtime import runtime_options


def wav():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    return buffer.getvalue()


class RuntimeSpy:
    loaded = False
    _loading_stage = None
    _resolved_device = None
    quantized_layers = 0
    paused = False
    last_error = None
    optimize = False
    compile_fallback_used = False

    def __init__(self):
        self.runtime_calls = []
        self.last_stats = {"seconds": 0.01, "audio_seconds": 0.1}
        self._runtime = {"device": "auto", "precision": "native", "cpu_threads": 0}

    def configure_runtime(self, options):
        # Match the real engine's normalization; the old spy stored options
        # directly and missed CPU/INT8 being reset to Auto/Native in engines.
        self._runtime = runtime_options(options)
        self.runtime_calls.append(dict(self._runtime))

    def generate_with_delivery(self, *args, **kwargs):
        return wav()

    def generate(self, *args, **kwargs):
        return wav()

    def unload(self):
        self.loaded = False

    def try_pause(self):
        return True

    def resume(self):
        return None


def test_vox_api_uses_saved_cpu_int8_then_test_only_override(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path / "voices"))
    monkeypatch.setattr(server, "KEY", "")
    spy = RuntimeSpy()
    monkeypatch.setattr(server, "ENGINE", spy)
    save_defaults(tmp_path, VOX_ID, {
        "runtime_device": "cpu", "runtime_precision": "int8",
        "runtime_cpu_threads": 8, "response_format": "wav",
    })
    client = TestClient(server.app)

    # Normal OpenAI-compatible API request inherits CPU/INT8 from the server.
    response = client.post("/v1/audio/speech", json={
        "model": VOX_ID, "input": "Use CPU by default", "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert spy.runtime_calls[-1] == {
        "device": "cpu", "precision": "int8", "cpu_threads": 8,
    }

    # Test pane can override one or both settings for this single request.
    response = client.post("/v1/audio/speech", json={
        "model": VOX_ID, "input": "Use CUDA for this one test",
        "runtime_device": "cuda", "runtime_precision": "native",
        "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert spy.runtime_calls[-1] == {
        "device": "cuda", "precision": "native", "cpu_threads": 8,
    }

    # Neither override mutates the persistent API defaults.
    assert effective_defaults(tmp_path, VOX_ID)["runtime_device"] == "cpu"
    assert effective_defaults(tmp_path, VOX_ID)["runtime_precision"] == "int8"
    response = client.post("/v1/audio/speech", json={
        "model": VOX_ID, "input": "Back to CPU", "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert spy.runtime_calls[-1]["device"] == "cpu"
    assert spy.runtime_calls[-1]["precision"] == "int8"


def test_qwen_api_uses_its_own_saved_defaults_not_vox(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path / "voices"))
    monkeypatch.setattr(server, "KEY", "")
    qwen = RuntimeSpy()
    monkeypatch.setattr(server, "QWEN_ENGINE", qwen)
    save_defaults(tmp_path, VOX_ID, {"runtime_device": "cuda"})
    save_defaults(tmp_path, QWEN_DEFAULT, {
        "runtime_device": "cpu", "runtime_precision": "int8",
        "response_format": "wav",
    })
    client = TestClient(server.app)
    result = client.post("/v1/audio/speech", json={
        "model": QWEN_DEFAULT, "input": "Hello", "voice": "serena",
        "response_format": "wav",
    })
    assert result.status_code == 200, result.text
    assert qwen.runtime_calls[-1] == {
        "device": "cpu", "precision": "int8", "cpu_threads": 0,
    }


def test_bad_runtime_overrides_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "KEY", "")
    client = TestClient(server.app)
    for key, value in [
        ("runtime_device", "rocm"), ("runtime_precision", "int4"),
    ]:
        response = client.post("/v1/audio/speech", json={
            "model": VOX_ID, "input": "Test", key: value,
        })
        assert response.status_code == 422, response.text


def test_runtime_status_reports_loading_without_sensitive_content(monkeypatch):
    spy = RuntimeSpy()
    spy._loading_stage = "Converting eligible VoxCPM2 layers to INT8 W8A8"
    spy._runtime = {"device": "cpu", "precision": "int8", "cpu_threads": 8}
    monkeypatch.setattr(server, "ENGINE", spy)
    data = TestClient(server.app).get("/health").json()
    assert data["loading"]["device"] == "cpu"
    assert data["loading"]["precision"] == "int8"
    assert data["loading"]["stage"] == "Converting eligible VoxCPM2 layers to INT8 W8A8"
    assert "input" not in data["loading"]


def test_indextts_api_inherits_saved_cpu_int8_without_test_override(
        tmp_path, monkeypatch):
    from tts_api.models import INDEX_ID
    monkeypatch.setattr(server, "ROOT", tmp_path)
    store = server.VoiceStore(tmp_path / "voices")
    recording = store.create(name="Reference", reference=wav())
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "KEY", "")
    index = RuntimeSpy()
    monkeypatch.setattr(server, "INDEX_ENGINE", index)
    save_defaults(tmp_path, INDEX_ID, {
        "voice": recording["id"], "runtime_device": "cpu",
        "runtime_precision": "int8", "runtime_cpu_threads": 8,
        "response_format": "wav",
    })
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": INDEX_ID, "input": "IndexTTS on CPU", "response_format": "wav",
    })
    assert response.status_code == 200, response.text
    assert index.runtime_calls[-1] == {
        "device": "cpu", "precision": "int8", "cpu_threads": 8,
    }
