"""CPU TTS uses its own authorized scope; Auto/CUDA always use ComfyUI."""
from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager

import anyio
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tts_api import comfy_queue, server
from tts_api.model_defaults import save_defaults
from tts_api.models import VOX_ID


def wav():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(4800, dtype=np.float32), 24000, format="WAV")
    return buffer.getvalue()


def test_cpu_scope_does_not_contact_queue_and_cannot_authorize_gpu(monkeypatch):
    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")

    class ForbiddenQueueClient:
        def __init__(self):
            raise AssertionError("CPU must not create queue client")

    monkeypatch.setattr(comfy_queue, "QueueClient", ForbiddenQueueClient)

    async def run():
        with pytest.raises(comfy_queue.QueueUnavailable):
            comfy_queue.require_queue_reservation("cpu")
        async with comfy_queue.cpu_tts("speech"):
            assert comfy_queue.active_request_kind() == "speech"
            await anyio.to_thread.run_sync(comfy_queue.require_queue_reservation, "cpu")
            for choice in ("auto", "cuda"):
                with pytest.raises(comfy_queue.QueueUnavailable):
                    comfy_queue.require_queue_reservation(choice)
                with pytest.raises(comfy_queue.QueueUnavailable):
                    await anyio.to_thread.run_sync(
                        comfy_queue.require_queue_reservation, choice
                    )
        assert comfy_queue.active_request_kind() is None
        with pytest.raises(comfy_queue.QueueUnavailable):
            comfy_queue.require_queue_reservation("cpu")

    asyncio.run(run())


def test_explicit_cpu_api_speech_and_design_skip_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path / "voices"))
    monkeypatch.setattr(server, "KEY", "")
    monkeypatch.setenv("TTS_COMFY_QUEUE_URL", "http://127.0.0.1:8188/tts-api/queue")
    monkeypatch.setenv("TTS_CONTROL_TOKEN", "test-secret")
    monkeypatch.setattr(server, "_render", lambda request: (
        comfy_queue.require_queue_reservation("cpu") or wav()
    ))
    monkeypatch.setattr(server, "_design_voice", lambda request: (
        comfy_queue.require_queue_reservation("cpu") or wav()
    ))

    @asynccontextmanager
    async def forbidden_queue(kind):
        raise AssertionError(f"CPU {kind} must not enter ComfyUI queue")
        yield

    monkeypatch.setattr(server, "queued_tts", forbidden_queue)
    save_defaults(tmp_path, VOX_ID, {
        "runtime_device": "cpu", "runtime_precision": "native",
        "response_format": "wav",
    })
    client = TestClient(server.app)
    speech = client.post("/v1/audio/speech", json={
        "model": VOX_ID, "input": "Hello", "response_format": "wav",
    })
    assert speech.status_code == 200, speech.text
    assert speech.content.startswith(b"RIFF")
    design = client.post("/api/voices/design", json={
        "model": VOX_ID, "name": "Narrator", "description": "Gentle",
        "sample_text": "Hello there",
    })
    assert design.status_code == 200, design.text


def test_test_tab_cpu_override_skips_but_auto_and_cuda_reserve(
        monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "KEY", "")
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path / "voices"))
    monkeypatch.setattr(server, "_render", lambda request: wav())
    calls = []

    @asynccontextmanager
    async def tracked_queue(kind):
        calls.append(kind)
        yield

    monkeypatch.setattr(server, "queued_tts", tracked_queue)
    save_defaults(tmp_path, VOX_ID, {
        "runtime_device": "cpu", "response_format": "wav",
    })
    client = TestClient(server.app)
    for override, expected in ((None, 0), ("cpu", 0),
                               ("auto", 1), ("cuda", 2)):
        payload = {"model": VOX_ID, "input": "Hello", "response_format": "wav"}
        if override is not None:
            payload["runtime_device"] = override
        result = client.post("/v1/audio/speech", json=payload)
        assert result.status_code == 200, result.text
        assert len(calls) == expected


def test_gpu_reclaim_keeps_loaded_cpu_model(monkeypatch):
    class FakeEngine:
        def __init__(self, device):
            self.loaded = True
            self._resolved_device = device
            self.unloads = 0

        def unload(self):
            self.unloads += 1
            self.loaded = False

    cpu = FakeEngine("cpu")
    gpu = FakeEngine("cuda:0")
    monkeypatch.setattr(server, "ENGINE", cpu)
    monkeypatch.setattr(server, "QWEN_ENGINE", gpu)
    index = FakeEngine("cpu")
    monkeypatch.setattr(server, "INDEX_ENGINE", index)

    assert server._reclaim_idle_gpu() == (True, True)
    assert cpu.loaded and index.loaded
    assert not gpu.loaded
    assert cpu.unloads == 0
    assert index.unloads == 0
    assert gpu.unloads == 1
    assert server._reclaim_idle_gpu() == (True, False)


def test_standalone_cpu_scope_without_bridge(monkeypatch):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)

    async def run():
        async with comfy_queue.cpu_tts("voice_design"):
            assert comfy_queue.active_request_kind() == "voice_design"
            comfy_queue.require_queue_reservation("cpu")

    asyncio.run(run())
