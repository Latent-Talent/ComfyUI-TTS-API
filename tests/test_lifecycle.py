"""Lifecycle regression tests. They never start a GPU model."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from managed_service import ManagedTTS
from tts_api.engine import VoxEngine
from tts_api.qwen_engine import QwenEngine
from tts_api import server


def test_auto_start_is_default_and_all_workflows_preempt_tts(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    triggered = []
    manager.reserve_for_comfy = lambda: triggered.append(True)
    manager.on_prompt({"prompt": {"1": {"class_type": "KSampler"}}})
    assert triggered == [True]
    assert not manager._disabled
    manager.on_prompt({"prompt": {}})
    assert triggered == [True]


def test_unresponsive_owned_worker_is_terminated(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    calls = []
    manager._running = lambda: True
    manager._request = lambda path, timeout=2.0: calls.append(path) or False
    manager._terminate = lambda: calls.append("terminated")
    manager.reserve_for_comfy()
    assert manager._reserved
    assert calls == ["/api/internal/reclaim", "terminated"]


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



def test_normal_comfy_workflow_reclaims_without_pausing_or_restarting(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    calls = []
    manager._running = lambda: True
    manager._request = lambda path, timeout=2.0: calls.append(path) or True
    manager._terminate = lambda: calls.append("terminate")
    manager._spawn = lambda: calls.append("spawn")
    manager.on_prompt({"prompt": {"1": {"class_type": "KSampler"}}})
    assert calls == ["/api/internal/reclaim"]
    assert manager._reserved


def test_pending_queue_does_not_start_unnecessary_pause_cycle(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    calls = []

    class TwoTicks:
        ticks = 0
        def wait(self, interval):
            self.ticks += 1
            return self.ticks > 1

    manager._stop = TwoTicks()
    manager._relay_events = lambda: None
    manager._queue_has_real_running = lambda: False
    manager._queue_busy = lambda: True  # Some workflow waiting in the queue
    manager._running = lambda: True
    manager._request = lambda path, timeout=2.0: calls.append(path) or True
    manager._terminate = lambda: calls.append("terminate")
    manager._spawn = lambda: calls.append("spawn")
    manager._watch_queue()
    assert calls == []
    assert not manager._reserved


def test_reclaim_keeps_idle_api_online_without_setting_paused(monkeypatch):
    monkeypatch.setattr(server, "CONTROL_TOKEN", "control-test-value")
    monkeypatch.setattr(server, "ENGINE", VoxEngine("test"))
    monkeypatch.setattr(server, "QWEN_ENGINE", QwenEngine())
    client = TestClient(server.app)
    headers = {"X-TTS-Control-Token": "control-test-value"}
    assert client.post("/api/internal/reclaim").status_code == 403
    response = client.post("/api/internal/reclaim", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"ready": True, "reclaimed": False, "paused": False}
    assert not server.ENGINE.paused
    assert not server.QWEN_ENGINE.paused
    # With no model cached there is no reason to pause or log a model unload.
    response2 = client.post("/api/internal/reclaim", headers=headers)
    assert response2.json() == response.json()


def test_reclaim_unloads_cached_model_but_does_not_pause_future_speech(monkeypatch):
    engine = VoxEngine("test")
    engine._model = object()
    monkeypatch.setattr(server, "CONTROL_TOKEN", "control-test-value")
    monkeypatch.setattr(server, "ENGINE", engine)
    monkeypatch.setattr(server, "QWEN_ENGINE", QwenEngine())
    response = TestClient(server.app).post(
        "/api/internal/reclaim",
        headers={"X-TTS-Control-Token": "control-test-value"},
    )
    assert response.status_code == 200
    assert response.json()["reclaimed"] is True
    assert not engine.loaded
    assert not engine.paused


def test_reclaim_reports_busy_without_terminating_or_pausing(monkeypatch):
    engine = VoxEngine("test")
    monkeypatch.setattr(server, "ENGINE", engine)
    monkeypatch.setattr(server, "QWEN_ENGINE", QwenEngine())
    monkeypatch.setattr(server, "CONTROL_TOKEN", "control-test-value")
    server.INFERENCE_LOCK.acquire()
    # RLock only blocks another thread; call the endpoint on a worker thread.
    try:
        response = TestClient(server.app).post(
            "/api/internal/reclaim",
            headers={"X-TTS-Control-Token": "control-test-value"},
        )
    finally:
        server.INFERENCE_LOCK.release()
    assert response.status_code == 409
    assert not engine.paused
