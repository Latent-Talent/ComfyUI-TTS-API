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
import re
import tempfile
import secrets
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path

LOG = logging.getLogger("comfyui-tts-api")
ROOT = Path(__file__).resolve().parent
ALLOWED_CONFIG = {
    "TTS_HOST", "TTS_PORT", "TTS_MODEL", "TTS_ACTIVE_MODEL", "TTS_API_KEY",
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
        # Track only our sanitized operations.log, never the raw worker log.
        self._event_offset = 0
        self._event_identity = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor = None
        self._reserved = False
        self._hold_until = 0.0
        self._installed = False
        self._owns_child = False
        self._disabled = self.config.get("TTS_AUTO_START", "true").lower() in ("0", "false", "no")
        self._skip_warning = False

    @property
    def api_key_configured(self):
        # Never return the key itself to a passive status request.
        return bool(self.config.get("TTS_API_KEY"))

    @property
    def api_key_editable(self):
        # Environment variables override .env for the managed process.
        return "TTS_API_KEY" not in os.environ

    def configure_api_key(self, action: str, requested_key: str | None = None):
        """Configure the key in .env and restart only our own managed worker.

        Only an explicit sidebar interaction may receive a generated key.
        The existing key is deliberately not recoverable through the UI.
        """
        if action not in ("generate", "set"):
            raise ValueError("Unknown key operation")
        with self._lock:
            if not self.api_key_editable:
                raise PermissionError("TTS_API_KEY is controlled by an environment variable")
            if self._queue_busy():
                raise RuntimeError("Cannot change the API key while a ComfyUI workflow is active")
            if not self._running() and self._port_open():
                raise RuntimeError("Cannot modify the key of an independently started service")

            key = secrets.token_urlsafe(32) if action == "generate" else requested_key
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9._~+\/=\-]{24,256}", key):
                raise ValueError("API key must be 24–256 safe ASCII characters")
            # Replace every existing assignment so an old duplicate cannot win.
            destination = self.root / ".env"
            old = destination.read_text(encoding="utf-8") if destination.exists() else ""
            lines = [line for line in old.splitlines()
                     if not re.match(r"^\s*(?:export\s+)?TTS_API_KEY\s*=", line)]
            lines.append("TTS_API_KEY=" + key)
            self.root.mkdir(parents=True, exist_ok=True)
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=str(self.root),
                    prefix=".tts-env-", delete=False
                ) as temp:
                    temp_name = temp.name
                    os.chmod(temp_name, 0o600)
                    temp.write("\n".join(lines) + "\n")
                    temp.flush()
                    os.fsync(temp.fileno())
                os.replace(temp_name, destination)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)

            self.config["TTS_API_KEY"] = key
            # An old worker would continue to accept the old key: restart it.
            was_running = self._running()
            if was_running:
                self._terminate()
            started = self._spawn() if not self._disabled else False
            return {
                "configured": True, "restarting": started, "managed": not self._disabled,
                "key": key if action == "generate" else None,
            }

    def configure_active_model(self, model_id: str) -> dict:
        """Persist default TTS backend and rotate only the owned child process."""
        # Import only lightweight model metadata, never CUDA, in ComfyUI.
        try:
            from .tts_api.models import canonical_model
        except ImportError:
            from tts_api.models import canonical_model
        if not isinstance(model_id, str):
            raise ValueError("Model ID must be a string")
        canonical = canonical_model(
            model_id, voxcpm_checkpoint=self.config.get("TTS_MODEL", "openbmb/VoxCPM2")
        )
        if canonical != model_id:
            raise ValueError("Select a canonical model ID from the model list")

        with self._lock:
            if "TTS_ACTIVE_MODEL" in os.environ:
                raise PermissionError("TTS_ACTIVE_MODEL is controlled by an environment variable")
            if self._queue_busy():
                raise RuntimeError("Cannot switch models while ComfyUI has queued workflows")
            if not self._running() and self._port_open():
                raise RuntimeError("Cannot configure an independently started service")
            if self.config.get("TTS_ACTIVE_MODEL", "voxcpm2") == canonical:
                return {"active_model": canonical, "restarting": False}

            destination = self.root / ".env"
            old = destination.read_text(encoding="utf-8") if destination.exists() else ""
            lines = [
                line for line in old.splitlines()
                if not re.match(r"^\s*(?:export\s+)?TTS_ACTIVE_MODEL\s*=", line)
            ]
            lines.append("TTS_ACTIVE_MODEL=" + canonical)
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=str(self.root),
                    prefix=".tts-env-", delete=False
                ) as fh:
                    temporary = fh.name
                    os.chmod(temporary, 0o600)
                    fh.write("\n".join(lines) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)

            self.config["TTS_ACTIVE_MODEL"] = canonical
            if self._running():
                self._terminate()
            started = self._spawn() if not self._disabled else False
            return {"active_model": canonical, "restarting": started}

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
        # ComfyUI may use cudaMallocAsync in its own process. VoxCPM2's
        # compiled CUDA graph path calls checkPoolLiveAllocations, which that
        # allocator cannot implement. Pin ONLY the child to native before
        # importing torch; do not touch ComfyUI's allocator configuration.
        environment["PYTORCH_ALLOC_CONF"] = "backend:native"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
        environment["TTS_CONTROL_TOKEN"] = self.control_token
        environment["TTS_PARENT_PID"] = str(os.getpid())
        environment["TTS_DATA_DIR"] = str(data_dir) if "TTS_DATA_DIR" not in self.config else self.config["TTS_DATA_DIR"]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        kwargs = {"cwd": str(self.root), "env": environment,
                  "stdin": subprocess.DEVNULL, "stdout": self._log_handle,
                  "stderr": subprocess.STDOUT, "creationflags": creationflags}
        if os.name == "posix":
            # No Python preexec_fn: forking with ComfyUI's many active threads
            # could otherwise deadlock. The service monitors TTS_PARENT_PID.
            kwargs["start_new_session"] = True
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
            # Skip events from prior ComfyUI sessions, but retain new events
            # written immediately after the managed worker is spawned.
            path = self.root / "data" / "operations.log"
            try:
                state = path.stat()
                self._event_identity = (state.st_dev, state.st_ino)
                self._event_offset = state.st_size
            except OSError:
                self._event_identity = None
                self._event_offset = 0
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

    def _relay_events(self):
        """Forward only allowlisted worker activity to the main ComfyUI console.

        The child service's stdout/stderr remain in managed-server.log because
        third-party libraries might include speech content in their diagnostics.
        Never mirror that raw stream here.
        """
        path = self.root / "data" / "operations.log"
        try:
            state = path.stat()
            identity = (state.st_dev, state.st_ino)
            if identity != self._event_identity or state.st_size < self._event_offset:
                self._event_offset = 0
                self._event_identity = identity
            with path.open("rb") as handle:
                handle.seek(self._event_offset)
                chunk = handle.read(65536)
            last_line = chunk.rfind(b"\n")
            if last_line < 0:
                return
            self._event_offset += last_line + 1
            try:
                from .tts_api import operations as activity
            except ImportError:
                # Tests/standalone use managed_service as a top-level module.
                from tts_api import operations as activity
            EVENTS = activity.EVENTS
            ID_PATTERN = activity.ID_PATTERN
            VALID_STRINGS = activity.VALID_STRINGS
            INT_FIELDS = activity.INT_FIELDS
            FLOAT_FIELDS = activity.FLOAT_FIELDS
            BOOL_FIELDS = activity.BOOL_FIELDS
            for line in chunk[:last_line + 1].splitlines():
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or record.get("event") not in EVENTS:
                        continue
                    pieces = [record["event"]]
                    rid = record.get("request_id")
                    if isinstance(rid, str) and ID_PATTERN.fullmatch(rid):
                        pieces.append("req=" + rid)
                    # This is an allowlist, NOT a generic dict dump. The
                    # message text, descriptions, exception strings, API keys,
                    # voice names and file paths are never forwarded.
                    for key in (
                        "model", "backend", "kind", "voice_type", "reason",
                        "status", "characters", "audio_seconds", "audio_bytes",
                        "elapsed_ms", "error_type", "cached",
                    ):
                        value = record.get(key)
                        valid = (
                            key in VALID_STRINGS and isinstance(value, str)
                            and VALID_STRINGS[key].fullmatch(value)
                        ) or (
                            key in INT_FIELDS and type(value) is int
                            and 0 <= value <= 10**12
                        ) or (
                            key in FLOAT_FIELDS and type(value) in (int, float)
                            and 0 <= value < 10**10
                        ) or (key in BOOL_FIELDS and type(value) is bool)
                        if valid:
                            pieces.append(f"{key}={value}")
                    LOG.info("[TTS API] %s", " ".join(pieces))
                except (ValueError, UnicodeDecodeError, TypeError):
                    continue
        except (OSError, ImportError):
            # Logging must not affect worker scheduling or speech generation.
            return

    def _watch_queue(self):
        # Do not import CUDA libraries or unload ComfyUI models from this thread.
        while not self._stop.wait(self.interval):
            self._relay_events()
            with self._lock:
                if self._disabled:
                    continue
                busy = self._queue_busy()
                if busy and not self._reserved:
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
        # This extension exposes a sidebar and a speech API, no workflow nodes.
        # Any submitted ComfyUI workflow gets the GPU ahead of background TTS.
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
            from .sidebar_routes import register_routes
            register_routes()
        else:
            LOG.warning("[TTS API] No PromptServer instance; sidebar routes and preemption unavailable")
    except (ImportError, AttributeError, RuntimeError) as exc:
        LOG.warning("[TTS API] Could not register ComfyUI integration: %s", exc)


atexit.register(_MANAGER.stop)
