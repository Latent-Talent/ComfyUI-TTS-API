"""Real ComfyUI TTS queue reservation tests without GPU or ComfyUI installed."""
import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from queue_bridge import QueueBroker, TTSQueueSlot
from managed_service import ManagedTTS
from tts_api import server
from tts_api.comfy_queue import QueueClient, QueueUnavailable


class FakePromptQueue:
    def __init__(self):
        self.queue = []
        self.running = []
        self.removed = []

    def put(self, item):
        self.queue.append(item)

    def get_current_queue_volatile(self):
        return list(self.running), list(self.queue)

    def delete_queue_item(self, predicate):
        for i, entry in enumerate(self.queue):
            if predicate(entry):
                self.removed.append(self.queue.pop(i))
                return True
        return False

    def start_next(self):
        item = self.queue.pop(0)
        self.running.append(item)
        return item


class FakePromptServer:
    def __init__(self):
        self.number = 2
        self.prompt_queue = FakePromptQueue()


class FakeManager:
    def __init__(self):
        self.calls = []
        self.alive = True

    def _running(self):
        return self.alive

    def _request(self, path, timeout=None):
        self.calls.append(("resume", path))
        return True

    def _terminate(self):
        self.calls.append(("terminate", None))
        self.alive = False


def new_ticket():
    return uuid.uuid4().hex


def test_queue_item_is_real_and_contains_no_speech_content():
    broker = QueueBroker()
    srv = FakePromptServer()
    ticket = new_ticket()
    prompt_id = broker.create(ticket, srv)
    queued = srv.prompt_queue.queue
    assert len(queued) == 1
    item = queued[0]
    assert item[0] == 2
    assert item[1] == prompt_id
    assert item[2]["1"]["class_type"] == "LatentTalentTTSQueueSlot"
    assert item[4] == ["1"]
    assert len(item) == 6
    assert "input" not in str(item)
    assert "reference_text" not in str(item)
    assert broker.state(ticket, srv.prompt_queue) == "pending"
    assert broker.outstanding()
    assert not broker.active()


def test_queue_node_blocks_running_slot_until_worker_releases():
    srv = FakePromptServer()
    broker = QueueBroker()
    manager = FakeManager()
    broker.configure(manager)
    ticket = new_ticket()
    broker.create(ticket, srv, kind="voice_design")
    srv.prompt_queue.start_next()
    result = []
    worker = threading.Thread(target=lambda: result.append(broker.execute(ticket)))
    worker.start()
    deadline = time.monotonic() + 2
    while broker.state(ticket, srv.prompt_queue) != "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert broker.active()
    assert worker.is_alive()
    assert broker.release(ticket)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result == [()]
    assert not broker.active()
    assert not broker.outstanding()
    assert ("resume", "/api/internal/resume") in manager.calls


def test_cancel_pending_comfyui_job_aborts_ticket_and_removes_reservation():
    srv = FakePromptServer()
    broker = QueueBroker()
    ticket = new_ticket()
    pid = broker.create(ticket, srv)
    assert srv.prompt_queue.delete_queue_item(lambda item: item[1] == pid)
    assert broker.state(ticket, srv.prompt_queue) == "cancelled"
    assert broker.release(ticket)
    assert not broker.outstanding()


def test_release_pending_ticket_removes_comfyui_queue_item():
    srv = FakePromptServer()
    broker = QueueBroker()
    ticket = new_ticket()
    broker.create(ticket, srv)
    assert broker.release(ticket)
    assert srv.prompt_queue.queue == []
    assert len(srv.prompt_queue.removed) == 1


def test_queue_worker_unavailable_must_not_start_generation():
    class FailingQueue(QueueClient):
        def _post(self, endpoint, ticket, kind="speech"):
            if endpoint == "acquire":
                raise QueueUnavailable("Simulated ComfyUI offline")
            raise AssertionError("Unexpected call")
    with pytest.raises(QueueUnavailable, match="offline"):
        FailingQueue(url="http://127.0.0.1:8188/tts-api/queue", token="test").acquire("speech")


def test_queue_client_waits_until_active_and_releases_when_finished():
    events = []

    class Client(QueueClient):
        def _post(self, endpoint, ticket, kind="speech"):
            events.append((endpoint, kind))
            if endpoint == "acquire":
                return {"state": "pending"}
            if endpoint == "state":
                return {"state": "pending" if events.count(("state", "speech")) == 1 else "running"}
            return {"released": True}

    client = Client(url="http://127.0.0.1:8188/tts-api/queue", token="test")
    ticket = client.acquire("speech")
    assert ticket
    client.release(ticket)
    assert events[0] == ("acquire", "speech")
    assert events[-1] == ("release", "speech")


def test_no_comfyui_queue_bridge_is_only_allowed_for_standalone():
    assert QueueClient(url="", token="").acquire("speech") is None
    with pytest.raises(QueueUnavailable, match="incomplete"):
        QueueClient(url="http://127.0.0.1:8188/tts-api/queue", token="").acquire("speech")


def test_workflow_submission_does_not_preempt_active_tts(monkeypatch, tmp_path):
    from queue_bridge import BROKER
    server_instance = FakePromptServer()
    ticket = new_ticket()
    pid = BROKER.create(ticket, server_instance)
    slot = BROKER._slots[ticket]
    slot.state = "running"
    manager = ManagedTTS(root=tmp_path)
    monkeypatch.setattr(manager, "_queue_has_real_running", lambda: False)
    manager.reserve_for_comfy()
    assert not manager._reserved
    BROKER.release(ticket)
    BROKER._slots.pop(ticket, None)


def test_speech_and_design_are_wrapped_in_queue_reservation(monkeypatch, tmp_path):
    events = []

    @asynccontextmanager
    async def track(kind):
        events.append("acquire " + kind)
        try:
            yield
        finally:
            events.append("release " + kind)

    monkeypatch.setattr(server, "queued_tts", track)
    monkeypatch.setattr(server, "_render", lambda request: b"RIFF" + b"0" * 100)
    monkeypatch.setattr(server, "_design_voice", lambda request: b"RIFF" + b"0" * 100)
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    client = TestClient(server.app)
    speech = client.post("/v1/audio/speech", json={
        "model": "voxcpm2", "input": "Hello",
        "response_format": "wav",
    })
    assert speech.status_code == 200
    design = client.post("/api/voices/design", json={
        "name": "Narrator", "description": "Warm voice", "sample_text": "Hello",
    })
    assert design.status_code == 200
    assert events == [
        "acquire speech", "release speech",
        "acquire voice_design", "release voice_design",
    ]
