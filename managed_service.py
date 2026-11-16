"""Lifecycle and GPU coordination for the optional ComfyUI-managed TTS service.

ComfyUI starts a lightweight API process, NOT the GPU model. On other workflow
submissions, the API unloads VoxCPM and pauses new inference. If it is busy and
cannot release promptly, the owned worker is terminated; it restarts after
ComfyUI becomes idle. The daemon watches the real ComfyUI prompt queue.

The managed child is never an arbitrary service discovered on the port.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path

LOG = logging.getLogger("comfyui-tts-api")
ROOT = Path(__file__).resolve().parent
TTS_CLASSES = {"ComfyTTSGenerate", "ComfyTTSCloneVoice", "ComfyTTSDesignVoice"}
ALLOWED_CONFIG = {
    "TTS_HOST", "TTS_PORT", "TTS_MODEL", "TTS_API_KEY",
    "TTS_DATA_DIR", "TTS_MAX_TEXT", "TTS_MAX_UPLOAD_MB",
    "TTS_UNLOAD_AFTER_REQUEST", "TTS_IDLE_UNLOAD_SECONDS", "TTS_AUTO_START",
}


def read_config(root=ROOT):
    """Parse the small .env subset without executing shell code."""
    values = {}
    path = root / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in ALLOWED_CONFIG:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                values[key] = value
    # Environment variables override .env.
    for key in ALLOWED_CONFIG:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


class ManagedTTS:
    def __init__(self, root=ROOT, interval=0.5):
        self.root = Path(root)
        self.interval = interval
        self.config = read_config(self.root)
        self.control_token = secrets.token_urlsafe(32)
        self.host = self.config.get("TTS_HOST", "127.0.0.1")
        self.port = int(self.config.get("TTS_PORT", "8025"))
        self.url = f"http://127.0.0.1:{self.port}"
        self._process = None
        self._log_handle = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor = None
        self._reserved = False
        self._hold_until = 0.0
        self._installed = False
        self._owns_child = False
        self._pure_tts_job = False
        self._disabled = self.config.get("TTS_AUTO_START", "true").lower() in ("0", "false", "no")
        self._skip_warning = False

    def _python(self):
        if os.name == "nt":
            return self.root / ".venv" / "Scripts" / "python.exe"
        return self.root / ".venv" / "bin" / "python"

    def _running(self):
        return self._process is not None and self._process.poll() is None

    def _request(self, endpoint: str, *, timeout=2.0):
        req = urllib.request.Request(
            self.url + endpoint,
            data=b"{}",
            headers={
                "X-TTS-Control-Token": self.control_token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status == 200
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
            return False

    def _healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=0.4) as response:
                return json.loads(response.read()).get("service") == "comfyui-tts-api"
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def is_local_url(self, url):
        try:
            parsed = urlsplit(url)
            return parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1") and (parsed.port or 80) == self.port
        except ValueError:
            return False

    def _port_open(self):
        import socket
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.25):
                return True
        except OSError:
            return False

    def _spawn(self):
        if self._disabled or self._running() or self._stop.is_set():
            return False
        executable = self._python()
        if not executable.is_file():
            if not self._skip_warning:
                LOG.warning(
                    "[TTS API] Managed server not installed; run 'bash install.sh' once in %s. "
                    "ComfyUI will start it automatically after restart.", self.root
                )
                self._skip_warning = True
            return False
        if self.host not in ("127.0.0.1", "localhost", "::1") and not self.config.get("TTS_API_KEY"):
            LOG.error("[TTS API] Refusing a non-loopback binding without TTS_API_KEY")
            return False
        if self._port_open():
            # Never seize or terminate a port belonging to another process.
            LOG.warning("[TTS API] Port %s already occupied; managed service not started", self.port)
            return False
        data_dir = self.root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (data_dir / "managed-server.log").open("ab", buffering=0)
        environment = os.environ.copy()
        environment.update(self.config)
        environment["TTS_CONTROL_TOKEN"] = self.control_token
        environment["TTS_DATA_DIR"] = str(data_dir) if "TTS_DATA_DIR" not in self.config else self.config["TTS_DATA_DIR"]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        kwargs = {"cwd": str(self.root), "env": environment,
                  "stdin": subprocess.DEVNULL, "stdout": self._log_handle,
                  "stderr": subprocess.STDOUT, "creationflags": creationflags}
        if os.name == "posix":
            kwargs["start_new_session"] = True
            if sys.platform.startswith("linux"):
                def on_parent_exit():
                    import ctypes
                    libc = ctypes.CDLL(None)
                    libc.prctl(1, signal.SIGTERM)
                    if os.getppid() == 1:
                        os._exit(1)
                kwargs["preexec_fn"] = on_parent_exit
        try:
            self._process = subprocess.Popen(
                [str(executable), "-m", "uvicorn", "tts_api.server:app",
                 "--host", self.host, "--port", str(self.port), "--workers", "1"],
                **kwargs,
            )
            self._owns_child = True
            LOG.info("[TTS API] Managed speech server started (PID %s), %s", self._process.pid, self.url)
            return True
        except (OSError, ValueError) as exc:
            LOG.error("[TTS API] Unable to start speech server: %s", exc)
            self._process = None
            return False
        finally:
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None

    def _terminate(self):
        proc = self._process
        self._process = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                        proc.wait(timeout=1.0)
                    except (OSError, subprocess.TimeoutExpired):
                        LOG.error("[TTS API] Unable to confirm worker termination")
        self._owns_child = False

    def start(self):
        with self._lock:
            self._spawn()
            if self._monitor is None and not self._disabled:
                self._monitor = threading.Thread(
                    target=self._watch_queue, name="TTS-ComfyUI-Memory-Coordinator", daemon=True
                )
                self._monitor.start()

    def _queue_busy(self):
        try:
            from server import PromptServer
            server = PromptServer.instance
            if server is None or server.prompt_queue is None:
                return False
            running, pending = server.prompt_queue.get_current_queue_volatile()
            return bool(running or pending)
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _watch_queue(self):
        # Do not import CUDA libraries or unload ComfyUI models from this thread.
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._disabled:
                    continue
                busy = self._queue_busy()
                if not busy:
                    self._pure_tts_job = False
                if busy and not self._reserved and not self._pure_tts_job:
                    self._reserved = True
                    self._hold_until = time.monotonic() + 1
                    self._yield_gpu()
                if (self._reserved and not busy and
                        time.monotonic() >= self._hold_until):
                    if not self._running():
                        self._spawn()
                    if self._running() and self._request("/api/internal/resume"):
                        self._reserved = False

    def _yield_gpu(self):
        """Must hold self._lock. Fail closed: stop OUR worker if it cannot unload."""
        if not self._running():
            return
        if not self._request("/api/internal/pause", timeout=2.0):
            LOG.warning("[TTS API] Worker busy/unresponsive; terminating to free ComfyUI VRAM")
            self._terminate()

    def reserve_for_comfy(self):
        with self._lock:
            self._reserved = True
            self._hold_until = time.monotonic() + 2.0
            self._yield_gpu()

    def on_prompt(self, payload):
        prompt = payload.get("prompt", {}) if isinstance(payload, dict) else {}
        if not isinstance(prompt, dict) or not prompt:
            return payload
        # Pure TTS workflow is permitted to use its own engine. Mixed workflows
        # yield first; TTS node executions temporarily resume the service.
        class_types = {v.get("class_type") for v in prompt.values() if isinstance(v, dict)}
        if class_types and class_types.issubset(TTS_CLASSES):
            with self._lock:
                if not self._reserved:
                    self._pure_tts_job = True
            return payload
        self._pure_tts_job = False
        self.reserve_for_comfy()
        return payload

    def allow_tts_node(self):
        with self._lock:
            if not self._running():
                self._spawn()
            if not self._running():
                if self._port_open():
                    # An independently started API may be usable, but we must
                    # never kill it or claim ownership of its VRAM.
                    return False
                raise RuntimeError("Managed TTS server unavailable; check data/managed-server.log")
            deadline = time.monotonic() + 15.0
            while not self._healthy() and time.monotonic() < deadline and self._running():
                time.sleep(0.15)
            if not self._healthy():
                raise RuntimeError("Managed TTS server did not become ready; inspect data/managed-server.log")
            if self._reserved:
                if not self._request("/api/internal/resume"):
                    raise RuntimeError("Cannot resume managed TTS server")
            return self._reserved

    def after_tts_node(self, yielded):
        if yielded:
            with self._lock:
                self._yield_gpu()

    def stop(self):
        self._stop.set()
        with self._lock:
            self._terminate()


_MANAGER = ManagedTTS()


def initialize():
    _MANAGER.start()
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            PromptServer.instance.add_on_prompt_handler(_MANAGER.on_prompt)
        else:
            LOG.warning("[TTS API] No PromptServer instance; queue preemption is unavailable")
    except (ImportError, AttributeError) as exc:
        LOG.warning("[TTS API] Unable to register ComfyUI queue hook: %s", exc)


def tts_node_call(fn, url):
    """Allow a ComfyUI TTS node to use the paused worker during mixed workflows."""
    if _MANAGER._disabled or not _MANAGER.is_local_url(url):
        return fn()
    yielded = _MANAGER.allow_tts_node()
    try:
        return fn()
    finally:
        _MANAGER.after_tts_node(yielded)


atexit.register(_MANAGER.stop)
