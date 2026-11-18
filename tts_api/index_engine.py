"""Optional IndexTTS 2.5 backend using its *separate Python 3.11 environment*.

The main OpenAI-compatible API can remain on Python 3.12. Only this backend's
weights and native dependencies reside in the IndexTTS subprocess. The worker
uses a private Unix-domain socket, never a public port. Spoken text is NEVER
printed into the ComfyUI operational log.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import soundfile as sf

from .comfy_queue import active_request_kind, require_queue_reservation
from .model_storage import index_checkpoints, index_python, index_source, model_root
from .models import INDEX_ID, INDEX_EMOTIONS
from .operations import emit

LOG = logging.getLogger("comfyui-tts-api.index")
MAX_IPC_BYTES = 2 * 1024 * 1024
INDEX_LANGUAGES = {
    "auto": "EN", "english": "EN", "en": "EN",
    "chinese": "ZH", "zh": "ZH",
    "japanese": "JA", "ja": "JA",
    "spanish": "ES", "es": "ES",
    "arabic": "AR", "ar": "AR",
}


class IndexEngine:
    def __init__(self, unload_after_request: bool = False):
        self._lock = threading.RLock()
        self._process = None
        self._connection = None
        self._reader = None
        self._tmp = None
        self._paused = False
        self._last_activity = time.monotonic()
        self.unload_after_request = unload_after_request
        self.last_stats: dict = {}
        self.last_error: dict | None = None
        self._runtime = {"device": "auto", "precision": "native", "cpu_threads": 0}
        self._resolved_device = None
        self.quantized_layers = 0
        self._loading_stage = None

    def configure_runtime(self, settings: dict) -> None:
        from .runtime import runtime_options
        config = runtime_options(settings)
        with self._lock:
            if config != self._runtime:
                self._unload_locked()
                self._runtime = config

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._connection is not None

    @property
    def paused(self) -> bool:
        return self._paused

    def _read(self) -> dict:
        line = self._reader.readline(MAX_IPC_BYTES)
        if not line or not line.endswith(b"\n"):
            raise RuntimeError("IndexTTS subprocess disconnected")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("Invalid IndexTTS subprocess reply")
        return value

    def _write(self, message: dict) -> None:
        self._connection.sendall((json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii"))

    def _unload_locked(self) -> None:
        loaded = self.loaded
        conn, reader, proc, temporary = self._connection, self._reader, self._process, self._tmp
        self._connection = self._reader = self._process = self._tmp = None
        self._resolved_device = None
        self.quantized_layers = 0
        self._loading_stage = None
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        if reader is not None:
            reader.close()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            else:
                proc.wait()
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)
        self._last_activity = time.monotonic()
        if loaded:
            emit("model.unloaded", backend="indextts", model=INDEX_ID)

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def try_pause(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._paused = True
            self._unload_locked()
            return True
        finally:
            self._lock.release()

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def unload_if_idle(self, max_idle_seconds: float) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if not self.loaded or time.monotonic() - self._last_activity < max_idle_seconds:
                return False
            self._unload_locked()
            return True
        finally:
            self._lock.release()

    def _load(self) -> None:
        require_queue_reservation()
        if self.loaded:
            return
        python = index_python()
        source = index_source()
        weights = index_checkpoints()
        if not python.is_file() or not source.is_dir():
            raise RuntimeError(
                "IndexTTS 2.5 backend not installed. Run bash install-index.sh "
                "inside ComfyUI-TTS-API and restart the TTS API."
            )
        if not (weights / "config.yaml").is_file():
            raise RuntimeError(
                "IndexTTS 2.5 weights not downloaded. Select IndexTTS and use "
                "Download weights in the TTS sidebar."
            )
        self._loading_stage = "Starting IndexTTS worker"
        emit("model.load.started", model=INDEX_ID, backend="indextts",
             kind=active_request_kind(), device=self._runtime["device"],
             precision=self._runtime["precision"])
        started = time.perf_counter()
        root = Path(tempfile.mkdtemp(prefix="comfy-tts-index-"))
        os.chmod(root, 0o700)
        sock_path = str(root / "rpc.sock")
        secret = secrets.token_hex(32)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.settimeout(1)
        log_file = None
        try:
            listener.bind(sock_path)
            listener.listen(1)
            data_dir = Path(os.getenv("TTS_DATA_DIR", "./data")).expanduser().resolve()
            data_dir.mkdir(parents=True, exist_ok=True)
            raw_log = data_dir / "index-worker.log"
            fd = os.open(raw_log, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            log_file = os.fdopen(fd, "ab", buffering=0)
            env = os.environ.copy()
            env.update({
                "TTS_INDEX_SOURCE": str(source),
                "TTS_INDEX_CHECKPOINTS": str(weights),
                "TTS_INDEX_SOCKET": sock_path,
                "TTS_INDEX_TOKEN": secret,
                "TTS_INDEX_OUTPUTS": str(root),
                "TTS_INDEX_PARENT_PID": str(os.getpid()),
                "TTS_INDEX_DEVICE": self._runtime["device"],
                "TTS_INDEX_PRECISION": self._runtime["precision"],
                "TTS_INDEX_CPU_THREADS": str(self._runtime["cpu_threads"]),
                "PYTHONPATH": os.pathsep.join([str(source), str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]),
            })
            self._process = subprocess.Popen(
                [str(python), "-u", "-m", "tts_api.index_worker"],
                cwd=str(source), env=env, stdin=subprocess.DEVNULL,
                stdout=log_file, stderr=subprocess.STDOUT,
            )
            self._tmp = str(root)
            # The subprocess and model can take a while to initialize.
            limit = time.monotonic() + 600
            connection = None
            while time.monotonic() < limit:
                if self._process.poll() is not None:
                    raise RuntimeError("IndexTTS model worker exited; see data/index-worker.log")
                try:
                    connection, _ = listener.accept()
                    break
                except socket.timeout:
                    continue
            if connection is None:
                raise TimeoutError("IndexTTS model initialization timed out")
            self._connection = connection
            self._connection.settimeout(600)
            self._reader = connection.makefile("rb")
            hello = self._read()
            if hello.get("event") != "hello" or not secrets.compare_digest(
                    str(hello.get("token", "")), secret):
                raise RuntimeError("IndexTTS worker authentication failed")
            self._loading_stage = "Loading IndexTTS model weights"
            ready = self._read()
            while ready.get("event") == "progress":
                stage = ready.get("stage")
                if stage in ("Loading IndexTTS model weights",
                             "Converting IndexTTS GPT to INT8 W8A8"):
                    self._loading_stage = stage
                    if stage == "Converting IndexTTS GPT to INT8 W8A8":
                        emit("model.quantization.started", backend="indextts",
                             device=self._runtime["device"], precision="int8")
                ready = self._read()
            if ready.get("event") != "ready":
                raise RuntimeError(
                    "IndexTTS initialization failed (" +
                    str(ready.get("error_type", "RuntimeError"))[:60] +
                    "); see data/index-worker.log"
                )
            self._resolved_device = ready.get("device")
            self.quantized_layers = int(ready.get("quantized_layers", 0))
            if self._runtime["precision"] == "int8":
                emit("model.quantization.completed", backend="indextts",
                     device=self._resolved_device, precision="int8",
                     quantized_layers=self.quantized_layers)
            self._loading_stage = None
            emit("model.load.completed", model=INDEX_ID, backend="indextts",
                 device=self._resolved_device, precision=self._runtime["precision"],
                 quantized_layers=self.quantized_layers,
                 elapsed_ms=(time.perf_counter() - started) * 1000)
        except BaseException as exc:
            if self._loading_stage == "Converting IndexTTS GPT to INT8 W8A8":
                emit("model.quantization.failed", backend="indextts",
                     precision="int8", error_type=type(exc).__name__)
            emit("model.load.failed", model=INDEX_ID, backend="indextts",
                 error_type=type(exc).__name__, elapsed_ms=(time.perf_counter() - started) * 1000)
            self._unload_locked()
            shutil.rmtree(root, ignore_errors=True)
            raise
        finally:
            listener.close()
            if log_file is not None:
                log_file.close()

    def generate(
        self, text: str, *, reference_path: str | None,
        language: str = "Auto", instructions: str = "",
        emotion_mode: str = "neutral", emotion_vector: list[float] | None = None,
        emotion_strength: float = 0.65, emotion_reference: str | None = None,
        duration_factor: float = 1.0, seed: int | None = None,
    ) -> bytes:
        require_queue_reservation()
        if not reference_path or not Path(reference_path).is_file():
            raise ValueError(
                "IndexTTS requires a saved voice recording. "
                "Upload/clone a voice first, then select it for speech."
            )
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        if emotion_mode not in ("neutral", "vector", "auto", "text", "reference"):
            raise ValueError("Unsupported IndexTTS emotion mode")
        if emotion_mode == "text" and not instructions.strip():
            raise ValueError("Emotion text mode requires delivery instructions")
        if emotion_mode == "reference" and (not emotion_reference or not Path(emotion_reference).is_file()):
            raise ValueError("Emotion reference mode requires a saved reference voice")
        if emotion_mode == "vector" and (
            not isinstance(emotion_vector, list) or len(emotion_vector) != len(INDEX_EMOTIONS)
            or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in emotion_vector)
        ):
            raise ValueError("IndexTTS emotion vector requires eight numbers between 0 and 1")
        lang = INDEX_LANGUAGES.get(language.strip().lower())
        if lang is None:
            raise ValueError("IndexTTS 2.5 supports English, Chinese, Japanese, Spanish, and Arabic")
        if not 0 <= emotion_strength <= 1 or not 0.5 <= duration_factor <= 2.0:
            raise ValueError("IndexTTS emotion strength or speaking duration out of range")
        with self._lock:
            if self._paused:
                raise RuntimeError("IndexTTS paused while ComfyUI needs GPU memory")
            started = time.perf_counter()
            self.last_error = None
            output_name = uuid.uuid4().hex + ".wav"
            try:
                self._load()
                self._connection.settimeout(1800)
                self._write({
                    "op": "generate", "text": text, "reference": reference_path,
                    "language": lang, "instructions": instructions,
                    "emotion_mode": emotion_mode,
                    "emotion_vector": emotion_vector,
                    "emotion_strength": emotion_strength,
                    "emotion_reference": emotion_reference,
                    "duration_factor": duration_factor, "seed": seed,
                    "output": output_name,
                })
                result = self._read()
                if result.get("event") != "completed":
                    raise RuntimeError(
                        "IndexTTS inference failed (" +
                        str(result.get("error_type", "RuntimeError"))[:60] +
                        "); see data/index-worker.log"
                    )
                path = Path(self._tmp) / output_name
                audio = path.read_bytes()
                with sf.SoundFile(io.BytesIO(audio)) as wav:
                    duration = wav.frames / wav.samplerate
                elapsed = time.perf_counter() - started
                self.last_stats = {
                    "seconds": round(elapsed, 3),
                    "audio_seconds": round(duration, 3),
                    "real_time_factor": round(elapsed / duration, 3) if duration else None,
                    "model": INDEX_ID,
                }
                return audio
            except BaseException as exc:
                self.last_error = {"stage": "index_generation",
                                   "type": type(exc).__name__,
                                   "message": str(exc)[:500]}
                emit("generation.failed", backend="indextts", model=INDEX_ID,
                     error_type=type(exc).__name__)
                # Crash or timeout can leave CUDA allocations in the child.
                self._unload_locked()
                raise
            finally:
                if self._tmp:
                    try:
                        (Path(self._tmp) / output_name).unlink(missing_ok=True)
                    except OSError:
                        pass
                self._last_activity = time.monotonic()
                if self.unload_after_request:
                    self._unload_locked()
