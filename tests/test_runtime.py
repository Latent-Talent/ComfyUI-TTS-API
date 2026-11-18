"""Runtime choices and selective TorchAO conversion; no torch installation needed."""
from __future__ import annotations

import sys
import types

import pytest

from tts_api.model_defaults import builtins_for, validate_settings
from tts_api.runtime import (
    apply_cpu_threads, quantize_int8, resolve_device, runtime_options,
)


def test_per_model_defaults_include_safe_runtime_choices():
    for model in ("voxcpm2", "indextts-2.5", "qwen3-tts-0.6b-customvoice"):
        values = builtins_for(model)
        assert values["runtime_device"] == "auto"
        assert values["runtime_precision"] == "native"
        assert values["runtime_cpu_threads"] == 0
        assert validate_settings(model, {
            "runtime_device": "cpu", "runtime_precision": "int8",
            "runtime_cpu_threads": 16,
        })["runtime_cpu_threads"] == 16
        with pytest.raises(ValueError):
            validate_settings(model, {"runtime_device": "tpu"})
        with pytest.raises(ValueError):
            validate_settings(model, {"runtime_precision": "int4"})
        with pytest.raises(ValueError):
            validate_settings(model, {"runtime_cpu_threads": 257})


def test_runtime_resolution_does_not_silently_ignore_gpu_request():
    def fake_torch(cuda):
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda)
        )

    assert resolve_device("auto", fake_torch(False)) == "cpu"
    assert resolve_device("cpu", fake_torch(True)) == "cpu"
    assert resolve_device("auto", fake_torch(True)) == "cuda:0"
    assert resolve_device("cuda", fake_torch(True)) == "cuda:0"
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        resolve_device("cuda", fake_torch(False))


def test_runtime_threads_are_explicit_and_bounded(monkeypatch):
    monkeypatch.setattr("tts_api.runtime.os.sched_getaffinity", lambda pid: set(range(12)),
                        raising=False)
    calls = []
    fake = types.SimpleNamespace(set_num_threads=calls.append)
    apply_cpu_threads(fake, 0, "cpu")
    apply_cpu_threads(fake, 8, "cuda:0")
    assert calls == []
    apply_cpu_threads(fake, 8, "cpu")
    assert calls == [8]
    with pytest.raises(ValueError, match="only 12"):
        apply_cpu_threads(fake, 16, "cpu")
    assert runtime_options({}) == {
        "device": "auto", "precision": "native", "cpu_threads": 0,
    }


class FakeLinear:
    def __init__(self, in_features=256, out_features=512):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = types.SimpleNamespace(numel=lambda: in_features * out_features)


def test_quantizer_only_selects_supported_talker_layers(monkeypatch):
    selected = []
    def quantize(model, config, filter_fn):
        selected.extend(name for name, layer in model.named_modules()
                        if filter_fn(layer, name))

    module = types.ModuleType("torchao.quantization")
    module.Int8DynamicActivationInt8WeightConfig = type("Config", (), {})
    module.quantize_ = quantize
    parent = types.ModuleType("torchao")
    parent.quantization = module
    monkeypatch.setitem(sys.modules, "torchao", parent)
    monkeypatch.setitem(sys.modules, "torchao.quantization", module)
    torch = types.SimpleNamespace(nn=types.SimpleNamespace(Linear=FakeLinear))
    layers = [
        ("talker.layers.0.attn.q_proj", FakeLinear()),
        ("talker.layers.0.norm", object()),
        ("codec.decoder.fc", FakeLinear()),
        ("talker.lm_head", FakeLinear()),
        ("talker.layers.0.tiny", FakeLinear(16, 16)),
    ]
    root = types.SimpleNamespace(named_modules=lambda: iter(layers))
    count = quantize_int8(root, scope="qwen", torch=torch)
    assert count == 1
    assert selected == ["talker.layers.0.attn.q_proj"]


def test_quantizer_fails_if_nothing_is_eligible(monkeypatch):
    module = types.ModuleType("torchao.quantization")
    module.Int8DynamicActivationInt8WeightConfig = type("Config", (), {})
    module.quantize_ = lambda *args, **kwargs: None
    parent = types.ModuleType("torchao")
    parent.quantization = module
    monkeypatch.setitem(sys.modules, "torchao", parent)
    monkeypatch.setitem(sys.modules, "torchao.quantization", module)
    torch = types.SimpleNamespace(nn=types.SimpleNamespace(Linear=FakeLinear))
    root = types.SimpleNamespace(named_modules=lambda: iter([
        ("codec.decoder", FakeLinear()),
    ]))
    with pytest.raises(RuntimeError, match="No compatible linear layers"):
        quantize_int8(root, scope="qwen", torch=torch)



@pytest.mark.parametrize("factory", [
    lambda: __import__("tts_api.engine", fromlist=["VoxEngine"]).VoxEngine("mock"),
    lambda: __import__("tts_api.qwen_engine", fromlist=["QwenEngine"]).QwenEngine(),
    lambda: __import__("tts_api.index_engine", fromlist=["IndexEngine"]).IndexEngine(),
])
def test_normalized_api_runtime_is_not_reset_by_engine(factory):
    """Catch the previous bug: API selects CPU/INT8, engine silently loads auto/native."""
    engine = factory()
    from_saved = runtime_options({
        "runtime_device": "cpu", "runtime_precision": "int8",
        "runtime_cpu_threads": 8,
    })
    assert from_saved == {"device": "cpu", "precision": "int8", "cpu_threads": 8}
    engine.configure_runtime(from_saved)
    assert engine._runtime == from_saved
    # Reapplying the normalized configuration must not change it.
    engine.configure_runtime(from_saved)
    assert engine._runtime == from_saved
    # Model API can also supply the saved-format configuration directly.
    engine.configure_runtime({
        "runtime_device": "cuda", "runtime_precision": "native",
        "runtime_cpu_threads": 12,
    })
    assert engine._runtime == {"device": "cuda", "precision": "native",
                               "cpu_threads": 12}


def test_runtime_options_reject_incomplete_or_conflicting_normalized_configuration():
    with pytest.raises(ValueError, match="Incomplete canonical"):
        runtime_options({"device": "cpu"})
    with pytest.raises(ValueError, match="Mixed canonical"):
        runtime_options({"device": "cpu", "precision": "int8", "cpu_threads": 8,
                         "runtime_device": "cuda"})
    with pytest.raises(ValueError, match="Unsupported TTS"):
        runtime_options({"device": "xpu", "precision": "int8", "cpu_threads": 8})
    with pytest.raises(ValueError, match="Unsupported TTS"):
        runtime_options({"device": "cpu", "precision": "int4", "cpu_threads": 8})
