"""ComfyUI-Manager exec restart must never produce two managed TTS workers."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from managed_service import ManagedTTS

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Inherited pipe and flock require POSIX/WSL"
)


def test_pipe_watchdog_exits_after_parent_fd_closes():
    """An exec restart closes this FD even when getppid() never changes."""
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env["TTS_PARENT_PIPE_FD"] = str(read_fd)
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "from tts_api.managed_entry import start_parent_watchdog; "
             "import time; start_parent_watchdog(); time.sleep(30)"],
            env=env, pass_fds=(read_fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        os.close(read_fd)
        read_fd = -1
        assert proc.poll() is None
        # Closing the CLOEXEC parent descriptor simulates Manager's os.execv.
        os.close(write_fd)
        write_fd = -1
        assert proc.wait(timeout=4) == 0
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_singleton_lock_persists_in_child_after_original_parent_closes(tmp_path):
    """Both manager instances cannot own the same managed worker at once."""
    data = tmp_path / "data"
    data.mkdir()
    first = ManagedTTS(root=tmp_path)
    second = ManagedTTS(root=tmp_path)
    fd = first._acquire_worker_lock(data)
    assert isinstance(fd, int)
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            pass_fds=(fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.close(fd)
        fd = -1
        assert second._acquire_worker_lock(data) is None
        # A second manager cannot start while the old worker still holds lock.
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
        (tmp_path / ".venv" / "bin" / "python").touch()
        second._port_open = lambda: False
        assert not second._spawn()
        assert second._process is None
    finally:
        if fd >= 0:
            os.close(fd)
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=4)
    renewed = second._acquire_worker_lock(data)
    assert renewed is not None
    os.close(renewed)


def test_monitor_recovers_worker_after_old_exec_child_finishes(tmp_path):
    manager = ManagedTTS(root=tmp_path)
    calls = []
    live = [False]

    class ThreeTicks:
        n = 0

        def wait(self, interval):
            self.n += 1
            return self.n > 3

    def spawn():
        calls.append("spawn")
        live[0] = True
        return True

    manager._stop = ThreeTicks()
    manager._relay_events = lambda: None
    manager._queue_has_real_running = lambda: False
    manager._running = lambda: live[0]
    manager._spawn = spawn
    manager._watch_queue()
    assert calls == ["spawn"]


def test_lock_is_not_retained_by_comfyui_after_spawn(tmp_path, monkeypatch):
    """Spawn receives the lock FD and the original parent closes its copy."""
    import managed_service
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    manager = ManagedTTS(root=tmp_path)
    manager._port_open = lambda: False
    captured = {}

    class StubProcess:
        pid = 99999
        def poll(self):
            return None

    def fake_spawn(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        captured["fds"] = kwargs.get("pass_fds")
        return StubProcess()

    monkeypatch.setattr(managed_service.subprocess, "Popen", fake_spawn)
    assert manager._spawn()
    assert captured["argv"][1:3] == ["-m", "tts_api.managed_entry"]
    assert len(captured["fds"]) == 2
    assert captured["env"]["TTS_PARENT_PIPE_FD"]
    assert captured["env"]["TTS_PARENT_PID"] == str(os.getpid())
    # Test process did not actually inherit the descriptors; clean the dummy
    # parent's write FD, without calling _terminate on the fake PID.
    if manager._parent_write_fd is not None:
        os.close(manager._parent_write_fd)
        manager._parent_write_fd = None
