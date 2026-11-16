"""Lifecycle regression tests. They never start a GPU model."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from managed_service import ManagedTTS
from tts_api.engine import VoxEngine
from tts_api import server


def test_auto_start_is_default_and_tts_only_prompts_are_not_preempted(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    triggered = []
    manager.reserve_for_comfy = lambda: triggered.append(True)
    manager.on_prompt({"prompt": {"1": {"class_type": "ComfyTTSGenerate"}}})
    assert manager._pure_tts_job
    assert not triggered
    manager.on_prompt({"prompt": {"1": {"class_type": "KSampler"}}})
    assert not manager._pure_tts_job
    assert triggered == [True]
    assert not manager._disabled


def test_unresponsive_owned_worker_is_terminated(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    calls = []
    manager._running = lambda: True
    manager._request = lambda path, timeout=2.0: False
    manager._terminate = lambda: calls.append("terminated")
    manager.reserve_for_comfy()
    assert manager._reserved
    assert calls == ["terminated"]


def test_local_url_detection(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    assert manager.is_local_url("http://127.0.0.1:8025")
    assert manager.is_local_url("http://localhost:8025/v1")
    assert not manager.is_local_url("http://127.0.0.1:9000")
    assert not manager.is_local_url("https://127.0.0.1:8025")
    assert not manager.is_local_url("http://example.com:8025")


def test_idle_unloading_and_pause_without_model_weights():
    engine = VoxEngine("not-loaded")
    engine._model = object()
    engine._last_activity = time.monotonic() - 120
    assert engine.unload_if_idle(60)
    assert not engine.loaded
    assert engine.try_pause()
    assert engine.paused
    engine.resume()
    assert not engine.paused


def test_internal_control_token(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "CONTROL_TOKEN", "test-control-token")
    engine = VoxEngine("not-loaded")
    monkeypatch.setattr(server, "ENGINE", engine)
    client = TestClient(server.app)
    assert client.post("/api/internal/pause").status_code == 403
    assert client.post("/api/internal/pause",
                       headers={"X-TTS-Control-Token": "bad"}).status_code == 403
    response = client.post("/api/internal/pause",
                           headers={"X-TTS-Control-Token": "test-control-token"})
    assert response.status_code == 200
    assert response.json()["paused"] is True
    response = client.post("/api/internal/resume",
                           headers={"X-TTS-Control-Token": "test-control-token"})
    assert response.status_code == 200
    assert response.json()["paused"] is False
