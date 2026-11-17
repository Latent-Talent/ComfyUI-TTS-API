"""CPU-only regression tests for GPU memory cleanup and multi-pass retention."""
from __future__ import annotations

import sys
from types import SimpleNamespace
import urllib.error

import numpy as np
import pytest
from fastapi.testclient import TestClient

from managed_service import ManagedTTS
from tts_api import engine as vox_module
from tts_api import qwen_engine as qwen_module
from tts_api import memory, server
from tts_api.engine import VoxEngine
from tts_api.qwen_engine import QwenEngine


@pytest.mark.parametrize("method", ["generate_with_delivery", "design_voice"])
def test_vox_two_pass_operation_reuses_model_before_unloading(monkeypatch, method):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)
    engine = VoxEngine("fake", unload_after_request=True)
    steps = []

    class Model:
        tts_model = SimpleNamespace(sample_rate=24000)
        def generate(self, **kwargs):
            steps.append("synthesize")
            return np.zeros(1200, dtype=np.float32)

    def load():
        if engine._model is None:
            steps.append("load")
            engine._model = Model()

    def unload():
        steps.append("unload")
        engine._model = None

    monkeypatch.setattr(engine, "_load", load)
    monkeypatch.setattr(engine, "_unload_locked", unload)
    if method == "generate_with_delivery":
        output = engine.generate_with_delivery("Hello", instructions="Speak gently")
    else:
        output = engine.design_voice("Hello", "Speak gently")
    assert output.startswith(b"RIFF")
    assert steps == ["load", "synthesize", "synthesize", "unload"]
    assert engine._retain_model_depth == 0


def test_vox_partial_model_load_cleans_allocator_without_gpu(monkeypatch):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)
    engine = VoxEngine("test")
    calls = []

    class StubVox:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("Simulated failure after some GPU allocations")

    monkeypatch.setitem(sys.modules, "voxcpm", SimpleNamespace(VoxCPM=StubVox))
    monkeypatch.setattr(vox_module, "release_unused_cuda", lambda: calls.append("cleanup"))
    with pytest.raises(RuntimeError, match="Simulated failure"):
        engine._load()
    assert not engine.loaded
    assert calls == ["cleanup"]


def test_qwen_partial_model_load_cleans_allocator_without_gpu(monkeypatch):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)
    engine = QwenEngine()
    calls = []

    class StubQwen:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("Simulated partial checkpoint failure")

    monkeypatch.setitem(sys.modules, "qwen_tts",
                        SimpleNamespace(Qwen3TTSModel=StubQwen))
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                        float32="float32"))
    monkeypatch.setattr(qwen_module, "release_unused_cuda", lambda: calls.append("cleanup"))
    with pytest.raises(RuntimeError, match="Simulated partial"):
        engine._load("qwen3-tts-1.7b-customvoice")
    assert not engine.loaded
    assert engine.loaded_model is None
    assert calls == ["cleanup"]


def test_cuda_memory_metrics_do_not_import_or_initialize_cuda(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    before = memory.process_cuda_memory()
    assert before == {"available": False, "reason": "torch_not_imported"}
    calls = []
    cuda = SimpleNamespace(
        is_initialized=lambda: False,
        device_count=lambda: calls.append("device_count"),
        empty_cache=lambda: calls.append("empty_cache"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert memory.process_cuda_memory() == {
        "available": False, "reason": "cuda_not_initialized"
    }
    memory.release_unused_cuda()
    assert calls == []


def test_cuda_memory_metrics_expose_allocated_and_reserved(monkeypatch):
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 1,
        memory_allocated=lambda device: 123456,
        memory_reserved=lambda device: 234567,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    result = memory.process_cuda_memory()
    assert result == {"available": True, "devices": [{
        "device": 0, "allocated_bytes": 123456,
        "reserved_bytes": 234567,
    }]}


def test_health_endpoint_includes_read_only_cuda_metrics(monkeypatch):
    monkeypatch.setattr(server, "process_cuda_memory",
                        lambda: {"available": True, "devices": [{
                            "device": 0, "allocated_bytes": 0, "reserved_bytes": 0,
                        }]})
    response = TestClient(server.app).get("/health")
    assert response.status_code == 200
    info = response.json()
    assert info["cuda_memory"]["devices"][0]["reserved_bytes"] == 0
    assert info["index_memory"]["separate_process"] is True


def test_reclaim_http_busy_is_classified_not_a_generic_failure(tmp_path, monkeypatch):
    import managed_service
    manager = ManagedTTS(root=tmp_path)
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("http://localhost", 409, "busy", {}, None)
    monkeypatch.setattr(managed_service.urllib.request, "urlopen", fail)
    assert manager._request_reclaim() == "busy"


@pytest.mark.parametrize("message,expected", [
    ("CUDA out of memory while allocating 400 MiB", True),
    ("torch.cuda.OutOfMemoryError: failed to allocate", True),
    ("Unsupported speaker ID", False),
])
def test_memory_exhaustion_detection(message, expected):
    assert memory.is_memory_exhaustion(RuntimeError(message)) is expected


def test_vox_inference_oom_drops_resident_checkpoint(monkeypatch):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)
    engine = VoxEngine("fake")

    class BrokenModel:
        def generate(self, **kwargs):
            raise RuntimeError("CUDA out of memory while allocating tensor")

    engine._model = BrokenModel()
    monkeypatch.setattr(engine, "_load", lambda: None)
    unloads = []

    def unload():
        unloads.append(True)
        engine._model = None

    monkeypatch.setattr(engine, "_unload_locked", unload)
    with pytest.raises(RuntimeError, match="out of memory"):
        engine.generate("Hello")
    assert not engine.loaded
    assert len(unloads) == 1


def test_qwen_inference_oom_drops_resident_checkpoint(monkeypatch):
    monkeypatch.delenv("TTS_COMFY_QUEUE_URL", raising=False)
    monkeypatch.delenv("TTS_CONTROL_TOKEN", raising=False)
    engine = QwenEngine()

    class BrokenModel:
        def generate_custom_voice(self, **kwargs):
            raise RuntimeError("CUDA out of memory while allocating tensor")

    engine._model = BrokenModel()
    engine._model_id = "qwen3-tts-1.7b-customvoice"
    monkeypatch.setattr(engine, "_load", lambda model_id: None)
    unloads = []

    def unload():
        unloads.append(True)
        engine._model = None
        engine._model_id = None

    monkeypatch.setattr(engine, "_unload_locked", unload)
    with pytest.raises(RuntimeError, match="out of memory"):
        engine.generate("qwen3-tts-1.7b-customvoice", "Hello")
    assert not engine.loaded
    assert len(unloads) == 1
