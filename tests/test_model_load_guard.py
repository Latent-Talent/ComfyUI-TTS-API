"""No managed TTS model may load without a running ComfyUI queue ticket."""
import asyncio

import anyio
import pytest

from tts_api.comfy_queue import (
    QueueUnavailable, active_request_kind, queued_tts,
    require_queue_reservation,
)
from tts_api.engine import VoxEngine
from tts_api.qwen_engine import QwenEngine


@pytest.mark.parametrize("backend", ["voxcpm2", "qwen3-tts"])
def test_managed_model_load_is_denied_without_reservation(monkeypatch, backend):
    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")
    engine = VoxEngine("unused") if backend == "voxcpm2" else QwenEngine()
    with pytest.raises(QueueUnavailable, match="active ComfyUI queue reservation"):
        if backend == "voxcpm2":
            engine._load()
        else:
            engine._load("qwen3-tts-1.7b-customvoice")
    assert not engine.loaded


def test_managed_queue_kind_propagates_to_inference_threads(monkeypatch):
    import tts_api.comfy_queue as queue
    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")
    releases = []

    class StubClient:
        def acquire(self, kind):
            assert kind == "speech"
            return "a" * 32

        def release(self, ticket, status="completed"):
            releases.append((ticket, status))

    monkeypatch.setattr(queue, "QueueClient", StubClient)

    async def run():
        with pytest.raises(QueueUnavailable):
            require_queue_reservation()
        async with queued_tts("speech"):
            assert active_request_kind() == "speech"
            await anyio.to_thread.run_sync(require_queue_reservation)
            kind = await anyio.to_thread.run_sync(active_request_kind)
            assert kind == "speech"
        assert active_request_kind() is None
        with pytest.raises(QueueUnavailable):
            require_queue_reservation()

    asyncio.run(run())
    assert releases == [("a" * 32, "completed")]


def test_failed_ticket_restores_guard_and_reports_failed(monkeypatch):
    import tts_api.comfy_queue as queue
    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")
    statuses = []

    class StubClient:
        def acquire(self, kind):
            return "b" * 32

        def release(self, ticket, status="completed"):
            statuses.append(status)

    monkeypatch.setattr(queue, "QueueClient", StubClient)

    async def run():
        with pytest.raises(RuntimeError, match="test failure"):
            async with queued_tts("voice_design"):
                assert active_request_kind() == "voice_design"
                await anyio.to_thread.run_sync(require_queue_reservation)
                raise RuntimeError("test failure")
        assert active_request_kind() is None
        with pytest.raises(QueueUnavailable):
            require_queue_reservation()

    asyncio.run(run())
    assert statuses == ["failed"]


def test_standalone_service_does_not_need_comfyui_reservation(monkeypatch):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)
    require_queue_reservation()


@pytest.mark.parametrize("backend", ["voxcpm2", "qwen3-tts"])
def test_managed_cached_inference_is_denied_without_reservation(monkeypatch, backend):
    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")
    engine = VoxEngine("unused") if backend == "voxcpm2" else QwenEngine()
    engine._model = object()  # simulates previously loaded weights, no CUDA
    with pytest.raises(QueueUnavailable, match="active ComfyUI queue reservation"):
        if backend == "voxcpm2":
            engine.generate("Unrequested input")
        else:
            engine.generate("qwen3-tts-1.7b-customvoice", "Unrequested input")
    assert engine.loaded


def test_model_load_event_identifies_authorized_request_kind(monkeypatch):
    import sys
    from types import SimpleNamespace
    import tts_api.comfy_queue as queue
    import tts_api.engine as vox_module

    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")

    class StubClient:
        def acquire(self, kind):
            return "c" * 32

        def release(self, ticket, status="completed"):
            pass

    monkeypatch.setattr(queue, "QueueClient", StubClient)
    fake_model = object()
    fake_vox = SimpleNamespace(VoxCPM=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: fake_model
    ))
    monkeypatch.setitem(sys.modules, "voxcpm", fake_vox)
    events = []
    monkeypatch.setattr(vox_module, "emit", lambda name, **kw: events.append((name, kw)))

    async def run():
        engine = VoxEngine("unused")
        async with queued_tts("voice_design"):
            await anyio.to_thread.run_sync(engine._load)
        assert engine.loaded

    asyncio.run(run())
    assert ("model.load.started", {
        "backend": "voxcpm2",
        "model": "voxcpm2",
        "kind": "voice_design",
    }) in events
