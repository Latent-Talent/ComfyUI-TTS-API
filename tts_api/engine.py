"""Lazy-loaded VoxCPM2 engine. Requires voxcpm in this service's own venv."""
from __future__ import annotations

import gc
import io
import logging
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import soundfile as sf

LOG = logging.getLogger("comfyui-tts-api.engine")

# These failure patterns point to compiled kernels/graph capture rather than
# invalid input or an ordinary GPU out-of-memory condition.
COMPILE_ERROR_MARKERS = (
    "checkPoolLiveAllocations",
    "cudaMallocAsync does not yet support",
    "BackendCompilerFailed",
    "CUDAGraphs only supports",
    "cudagraph capture",
    "torch._inductor.exc",
    "torch._dynamo.exc",
)


class VoxEngine:
    def __init__(self, model_name: str, unload_after_request: bool = False):
        self.model_name = model_name
        self.unload_after_request = unload_after_request
        self._model = None
        self._lock = threading.RLock()
        self.last_stats: dict = {}
        self.last_error: dict | None = None
        self.optimize = os.getenv("TTS_TORCH_COMPILE", "true").strip().lower() not in (
            "0", "false", "no", "off"
        )
        self.compile_fallback_used = False
        self._paused = False
        self._last_activity = time.monotonic()

    @property
    def paused(self) -> bool:
        return self._paused

    def _unload_locked(self) -> None:
        model = self._model
        self._model = None
        del model
        gc.collect()
        # The TTS backend is isolated from ComfyUI, so empty THIS process'
        # CUDA allocator; ComfyUI's allocator cannot free another PID's memory.
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._last_activity = time.monotonic()

    def try_pause(self) -> bool:
        """Nonblocking. Parent may terminate this worker if inference is active."""
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
            if self._model is None or time.monotonic() - self._last_activity < max_idle_seconds:
                return False
            self._unload_locked()
            return True
        finally:
            self._lock.release()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is None:
            try:
                from voxcpm import VoxCPM
            except ImportError as exc:
                raise RuntimeError("Missing VoxCPM engine. Run: pip install voxcpm") from exc
            self._model = VoxCPM.from_pretrained(
                self.model_name, load_denoiser=False, optimize=self.optimize
            )

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def generate(self, text: str, reference_path: Optional[str] = None,
                 reference_text: Optional[str] = None, description: Optional[str] = None,
                 seed: Optional[int] = None, steps: int = 10, cfg: float = 2.0) -> bytes:
        """Generate a complete WAV in memory; serializes model operations to avoid CUDA races."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        with self._lock:
            if self._paused:
                raise RuntimeError("TTS is paused while ComfyUI needs GPU memory")
            self.last_error = None
            try:
                self._load()
            except Exception as exc:
                self.last_error = {"stage": "model_load", "type": type(exc).__name__,
                                   "message": str(exc)[:1000]}
                LOG.exception("[TTS API] Failed to load VoxCPM2")
                raise
            self._last_activity = time.monotonic()
            kwargs = {"text": f"({description}){text}" if description else text,
                      "cfg_value": cfg, "inference_timesteps": steps}
            if seed is not None:
                kwargs["seed"] = seed
            if reference_path:
                kwargs["reference_wav_path"] = reference_path
            if reference_text:
                kwargs["prompt_text"] = reference_text
                kwargs["prompt_wav_path"] = reference_path
            start = time.perf_counter()
            try:
                try:
                    signal = self._model.generate(**kwargs)
                except Exception as exc:
                    # VoxCPM2 defaults to torch.compile(..., mode="reduce-overhead").
                    # Recover once from a known compile/graph incompatibility without
                    # disabling acceleration for healthy installations.
                    compiler_error = any(marker.lower() in str(exc).lower()
                                         or marker.lower() in type(exc).__name__.lower()
                                         for marker in COMPILE_ERROR_MARKERS)
                    if not self.optimize or not compiler_error:
                        raise
                    LOG.warning("[TTS API] Optimized inference failed (%s). "
                                "Retrying this request once without torch.compile.", exc)
                    self.optimize = False
                    self.compile_fallback_used = True
                    self._unload_locked()
                    self._load()
                    signal = self._model.generate(**kwargs)
                sample_rate = int(self._model.tts_model.sample_rate)
                audio = np.asarray(signal).squeeze()
                if audio.ndim != 1 or audio.size == 0:
                    raise RuntimeError("Model returned invalid audio shape.")
                buf = io.BytesIO()
                sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
                duration = len(audio) / sample_rate
                elapsed = time.perf_counter() - start
                self.last_stats = {"seconds": round(elapsed, 3), "audio_seconds": round(duration, 3),
                                   "real_time_factor": round(elapsed / duration, 3) if duration else None}
                return buf.getvalue()
            except Exception as exc:
                self.last_error = {"stage": "generation", "type": type(exc).__name__,
                                   "message": str(exc)[:1000]}
                LOG.exception("[TTS API] Speech generation failed")
                raise
            finally:
                self._last_activity = time.monotonic()
                if self.unload_after_request:
                    self._unload_locked()
