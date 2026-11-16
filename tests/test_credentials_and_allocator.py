"""API-key persistence and TTS allocator isolation tests (no GPU required)."""
from __future__ import annotations

import os
import stat

import managed_service
from managed_service import ManagedTTS, read_config


def test_generate_key_persists_without_leaking_existing_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    manager = ManagedTTS(root=tmp_path)
    manager._queue_busy = lambda: False
    manager._port_open = lambda: False
    manager._spawn = lambda: True
    (tmp_path / ".env").write_text(
        "# local settings\nTTS_PORT=8025\nTTS_API_KEY=oldsecret\n"
        "TTS_API_KEY=anotheroldsecret\n", encoding="utf-8"
    )

    result = manager.configure_api_key("generate")
    assert result["configured"] is True
    assert result["restarting"] is True
    key = result["key"]
    assert len(key) >= 24
    assert manager.api_key_configured is True
    assert read_config(tmp_path)["TTS_API_KEY"] == key
    contents = (tmp_path / ".env").read_text(encoding="utf-8")
    assert contents.count("TTS_API_KEY=") == 1
    assert "oldsecret" not in contents
    assert "TTS_PORT=8025" in contents
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_manual_key_rotation_and_validation(tmp_path, monkeypatch):
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    manager = ManagedTTS(root=tmp_path)
    manager._queue_busy = lambda: False
    manager._port_open = lambda: False
    manager._spawn = lambda: True
    key = "user-generated-very-long-key-value"
    result = manager.configure_api_key("set", key)
    assert result["configured"] is True
    assert result["key"] is None
    assert manager.config["TTS_API_KEY"] == key
    try:
        manager.configure_api_key("set", "too-short")
    except ValueError:
        pass
    else:
        raise AssertionError("Short API key unexpectedly accepted")
    assert read_config(tmp_path)["TTS_API_KEY"] == key


def test_environment_variable_api_key_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_API_KEY", "configured-outside-comfy")
    manager = ManagedTTS(root=tmp_path)
    assert not manager.api_key_editable
    try:
        manager.configure_api_key("generate")
    except PermissionError:
        pass
    else:
        raise AssertionError("Managed environment key should be immutable")
    assert not (tmp_path / ".env").exists()


def test_managed_worker_uses_native_allocator_without_changing_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("test", encoding="utf-8")
    manager = ManagedTTS(root=tmp_path)
    manager._port_open = lambda: False
    captured = {}

    class FakeProcess:
        pid = 10001
        def poll(self):
            return None

    def fake_spawn(argv, **kwargs):
        captured.update({"argv": argv, "env": kwargs["env"]})
        return FakeProcess()

    monkeypatch.setattr(managed_service.subprocess, "Popen", fake_spawn)
    assert manager._spawn()
    env = captured["env"]
    assert env["PYTORCH_ALLOC_CONF"] == "backend:native"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "backend:cudaMallocAsync"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"
