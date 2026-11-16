"""VoxCPM seed compatibility tests: do not require model weights or CUDA."""
from contextlib import contextmanager
import random
import sys
import types

import numpy as np
import pytest

from tts_api.engine import VoxEngine


class LegacyVox:
    def __init__(self):
        self.tts_model = types.SimpleNamespace(sample_rate=48000)
        self.received = []

    def _generate(self, text, cfg_value=2.0, inference_timesteps=10, reference_wav_path=None):
        self.received.append(text)
        return np.random.standard_normal(4800).astype(np.float32) * 0.01

    def generate(self, **kwargs):
        return self._generate(**kwargs)


class CurrentVox(LegacyVox):
    def _generate(self, text, cfg_value=2.0, inference_timesteps=10, seed=None):
        self.received.append(seed)
        return np.zeros(4800, dtype=np.float32)


def test_legacy_voxcpm_seed_reproducible_and_rng_state_restored(monkeypatch):
    torch_seeds = []

    @contextmanager
    def fake_fork_rng(devices):
        assert devices == []
        yield

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        random=types.SimpleNamespace(fork_rng=fake_fork_rng),
        manual_seed=lambda seed: torch_seeds.append(seed),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    model = LegacyVox()
    engine = VoxEngine("fake")
    engine._model = model
    np.random.seed(1234)
    random.seed(5678)
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    one = engine.generate("Hello", seed=42)
    two = engine.generate("Hello", seed=42)

    assert one == two
    assert torch_seeds == [42, 42]
    assert model.received == ["Hello", "Hello"]
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert random.getstate() == python_before


def test_current_voxcpm_receives_seed_directly():
    engine = VoxEngine("fake")
    model = CurrentVox()
    engine._model = model
    result = engine.generate("Hello", seed=73)
    assert result.startswith(b"RIFF")
    assert model.received == [73]


def test_uninspectable_version_retries_only_unsupported_seed(monkeypatch):
    @contextmanager
    def fake_fork_rng(devices):
        yield

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        random=types.SimpleNamespace(fork_rng=fake_fork_rng),
        manual_seed=lambda seed: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class OpaqueVox(LegacyVox):
        @property
        def _generate(self):
            raise AttributeError("_generate hidden in this release")

        def generate(self, **kwargs):
            if "seed" in kwargs:
                raise TypeError("VoxCPM._generate() got an unexpected keyword argument 'seed'")
            return np.zeros(4800, dtype=np.float32)

    engine = VoxEngine("fake")
    engine._model = OpaqueVox()
    assert engine.generate("Hello", seed=55).startswith(b"RIFF")


def test_unrelated_type_error_is_not_swallowed(monkeypatch):
    class BrokenVox(LegacyVox):
        def _generate(self, text, cfg_value=2.0, inference_timesteps=10, seed=None):
            raise TypeError("bad tensor shape")

    engine = VoxEngine("fake")
    engine._model = BrokenVox()
    with pytest.raises(TypeError, match="bad tensor shape"):
        engine.generate("Hello", seed=42)
    assert engine.last_error["type"] == "TypeError"


def test_voice_design_api_with_legacy_voxcpm(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from tts_api import server

    @contextmanager
    def fake_fork_rng(devices):
        yield

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        random=types.SimpleNamespace(fork_rng=fake_fork_rng),
        manual_seed=lambda seed: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path))
    engine = VoxEngine("legacy-model")
    engine._model = LegacyVox()
    monkeypatch.setattr(server, "ENGINE", engine)
    response = TestClient(server.app).post("/api/voices/design", json={
        "name": "Narrator",
        "description": "Deep, calm adult male voice.",
        "sample_text": "Hello, this is my new voice.",
        "seed": 42,
    })
    assert response.status_code == 200, response.text
    voice = response.json()
    assert voice["name"] == "Narrator"
    assert server.STORE.audio_path(voice).exists()
    # The only final sample synthesis input is the exact sample_text.
    # Voice instructions are confined to an unheard scratch reference.
    assert engine._model.received == [
        "(Deep, calm adult male voice.)Hello there.",
        "Hello, this is my new voice.",
    ]
