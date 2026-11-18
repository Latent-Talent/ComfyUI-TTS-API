"""Backend-level device and INT8 routing; no torch, CUDA or model download."""
from __future__ import annotations

import sys
import types


def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.bfloat16 = "bf16"
    torch.float32 = "fp32"
    torch.set_num_threads = lambda count: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_voxcpm_explicit_cpu_reaches_loader_and_triggers_int8(monkeypatch):
    from tts_api import engine as module
    torch = fake_torch(monkeypatch)
    calls = []
    core = types.SimpleNamespace(named_modules=lambda: iter(()))
    voxcpm = types.ModuleType("voxcpm")
    voxcpm.VoxCPM = types.SimpleNamespace(from_pretrained=lambda *a, **k:
        (calls.append(("load", a, k)) or types.SimpleNamespace(tts_model=core)))
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm)
    monkeypatch.setattr(module, "require_queue_reservation", lambda: None)
    monkeypatch.setattr(module, "release_unused_cuda", lambda: None)
    monkeypatch.setattr(module, "require_int8_backend", lambda: None)
    monkeypatch.setattr(module, "quantize_int8", lambda target, **kwargs:
                        (calls.append(("quant", target, kwargs)) or 37))
    monkeypatch.setattr(module, "emit", lambda event, **kw:
                        calls.append(("event", event, kw)))

    engine = module.VoxEngine("fake", unload_after_request=False)
    engine.configure_runtime({
        "runtime_device": "cpu", "runtime_precision": "int8",
        "runtime_cpu_threads": 1,
    })
    engine._load()
    load = next(call for call in calls if call[0] == "load")
    assert load[2]["device"] == "cpu"
    assert load[2]["optimize"] is False
    assert ("quant", core, {"scope": "voxcpm", "torch": torch}) in calls
    assert engine._resolved_device == "cpu"
    assert engine.quantized_layers == 37
    assert engine._loading_stage is None
    events = [event for _, event, kw in calls if _ == "event"]
    assert "model.load.completed" in events


def test_qwen_explicit_cpu_reaches_loader_and_triggers_int8(monkeypatch):
    from tts_api import qwen_engine as module
    from tts_api.models import QWEN_DEFAULT
    torch = fake_torch(monkeypatch)
    calls = []
    target = types.SimpleNamespace(named_modules=lambda: iter(()))
    wrapper = types.SimpleNamespace(model=target)
    qwen = types.ModuleType("qwen_tts")
    qwen.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=lambda *a, **k:
        (calls.append(("load", a, k)) or wrapper))
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen)
    monkeypatch.setattr(module, "require_queue_reservation", lambda: None)
    monkeypatch.setattr(module, "release_unused_cuda", lambda: None)
    monkeypatch.setattr(module, "require_int8_backend", lambda: None)
    monkeypatch.setattr(module, "quantize_int8", lambda t, **kwargs:
                        (calls.append(("quant", t, kwargs)) or 18))
    monkeypatch.setattr(module, "emit", lambda event, **kw: None)

    engine = module.QwenEngine()
    engine.configure_runtime({
        "runtime_device": "cpu", "runtime_precision": "int8",
        "runtime_cpu_threads": 1,
    })
    engine._load(QWEN_DEFAULT)
    load = next(call for call in calls if call[0] == "load")
    assert load[2]["device_map"] == "cpu"
    assert load[2]["dtype"] == "fp32"
    assert ("quant", target, {"scope": "qwen", "torch": torch}) in calls
    assert engine._resolved_device == "cpu"
    assert engine.quantized_layers == 18
    assert engine._loading_stage is None
