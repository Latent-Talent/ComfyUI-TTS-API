"""Lazy-loaded VoxCPM2 engine. Requires voxcpm in this service's own venv."""
from __future__ import annotations

import gc
from collections import OrderedDict
import inspect
import io
import random
from contextlib import contextmanager
import logging
import os
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import soundfile as sf

from .operations import emit
from .comfy_queue import active_request_kind, require_queue_reservation

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
        self._style_cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._paused = False
        self._last_activity = time.monotonic()

    @property
    def paused(self) -> bool:
        return self._paused

    def _unload_locked(self) -> None:
        previously_loaded = self._model is not None
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
        if previously_loaded:
            emit("model.unloaded", backend="voxcpm2", model="voxcpm2")

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
            require_queue_reservation()
            emit("model.load.started", backend="voxcpm2", model="voxcpm2",
                 kind=active_request_kind())
            load_started = time.perf_counter()
            try:
                from voxcpm import VoxCPM
                self._model = VoxCPM.from_pretrained(
                    self.model_name, load_denoiser=False, optimize=self.optimize
                )
            except Exception as exc:
                emit("model.load.failed", backend="voxcpm2",
                     model="voxcpm2", error_type=type(exc).__name__,
                     elapsed_ms=(time.perf_counter() - load_started) * 1000)
                raise
            emit("model.load.completed", backend="voxcpm2", model="voxcpm2",
                 elapsed_ms=(time.perf_counter() - load_started) * 1000)

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    @staticmethod
    def _accepts_seed(model) -> bool | None:
        """Inspect the internal VoxCPM method, not generate(*args, **kwargs).

        Some released voxcpm versions lack seed on _generate even though the
        latest upstream has it. None means the signature could not be read.
        """
        try:
            generate_impl = getattr(model, "_generate", None)
        except AttributeError:
            return None
        if generate_impl is None:
            return None
        try:
            parameters = inspect.signature(generate_impl).parameters
        except (TypeError, ValueError):
            return None
        return ("seed" in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ))

    @staticmethod
    @contextmanager
    def _legacy_seed(seed: int | None):
        """Best-effort seeded generation for VoxCPM releases without seed kwarg.

        Preserve Torch, NumPy and Python RNG states in the isolated worker.
        This doesn't promise bit-identical audio across different backends.
        """
        if seed is None:
            yield
            return
        import torch
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                np.random.seed(int(seed) % (2 ** 32))
                random.seed(int(seed))
                yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def _generate_compatible(self, kwargs: dict, seed: int | None):
        """Run current and older VoxCPM versions without dropping the seed."""
        call_kwargs = dict(kwargs)
        if seed is not None and self._accepts_seed(self._model) is False:
            call_kwargs.pop("seed", None)
            with self._legacy_seed(seed):
                return self._model.generate(**call_kwargs)
        try:
            return self._model.generate(**call_kwargs)
        except TypeError as exc:
            # Only retry when the error is specifically from a rejected seed
            # argument, not from arbitrary TypeErrors inside the TTS model.
            message = str(exc)
            if (seed is None or "seed" not in call_kwargs
                    or "unexpected keyword argument 'seed'" not in message):
                raise
            LOG.info("[TTS API] Installed VoxCPM does not accept seed; using "
                     "isolated RNG seeding instead")
            call_kwargs.pop("seed")
            with self._legacy_seed(seed):
                return self._model.generate(**call_kwargs)

    def generate_with_delivery(
        self, text: str, instructions: str = "",
        reference_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        seed: Optional[int] = None, steps: int = 10, cfg: float = 2.0,
    ) -> bytes:
        """Apply delivery instructions without injecting them into spoken text.

        VoxCPM2 only accepts delivery guidance as a parenthesized text prefix.
        First synthesize a short *unheard* style reference with that prefix,
        then clone its style while speaking the exact supplied text. Cache
        short references by settings to avoid a second generation each time.

        This is best-effort style conditioning, not a separate native control
        channel. The extra first pass is incurred once per distinct style.
        """
        style = instructions.strip()
        if not style:
            return self.generate(
                text, reference_path=reference_path,
                reference_text=reference_text, seed=seed, steps=steps, cfg=cfg,
            )
        if not text or not text.strip():
            raise ValueError("A non-empty text is required")
        if len(style) > 1000:
            raise ValueError("Delivery instructions must be 1000 characters or fewer")
        with self._lock:
            source_version = None
            if reference_path:
                info = os.stat(reference_path)
                source_version = (info.st_mtime_ns, info.st_size)
            key = (self.model_name, style, reference_path, source_version,
                   reference_text, seed, steps, cfg)
            scratch_audio = self._style_cache.get(key)
            if scratch_audio is None:
                emit("style.reference.started", backend="voxcpm2", cached=False)

                scratch_audio = self.generate(
                    "Hello there.", reference_path=reference_path,
                    reference_text=reference_text, description=style,
                    seed=seed, steps=steps, cfg=cfg,
                )
                self._style_cache[key] = scratch_audio
                if len(self._style_cache) > 12:
                    self._style_cache.popitem(last=False)
                emit("style.reference.completed", backend="voxcpm2", cached=False)
            else:
                self._style_cache.move_to_end(key)
                emit("style.reference.cached", backend="voxcpm2", cached=True)
            scratch_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix="voxcpm-style-", suffix=".wav",
                    delete=False,
                ) as tmp:
                    scratch_path = tmp.name
                    tmp.write(scratch_audio)
                return self.generate(
                    text, reference_path=scratch_path,
                    seed=seed, steps=steps, cfg=cfg,
                )
            finally:
                if scratch_path is not None:
                    try:
                        os.unlink(scratch_path)
                    except FileNotFoundError:
                        pass

    def design_voice(self, sample_text: str, description: str,
                     seed: int | None = None, steps: int = 10,
                     cfg: float = 2.0) -> bytes:
        """Create a described voice, then synthesize ONLY sample_text.

        VoxCPM2 accepts voice instructions exclusively as an inline
        '(description)' prefix. Some checkpoints speak those instructions.
        The first, short generated utterance therefore serves *only* as a
        timbre reference and is never returned as the designed voice sample.
        The second pass uses reference-only cloning with just sample_text.

        Two passes happen only when creating a reusable voice; ordinary
        playback stays single-pass and never injects the description.
        """
        if not sample_text or not sample_text.strip():
            raise ValueError("A non-empty sample text is required")
        if not description or not description.strip():
            raise ValueError("A non-empty voice description is required")
        clean_description = description.strip().replace("(", "").replace(")", "")
        if not clean_description:
            raise ValueError("Voice description cannot consist only of parentheses")
        with self._lock:
            # Never play or persist the scratch conditioning utterance.
            scratch_audio = self.generate(
                "Hello there.", description=clean_description,
                seed=seed, steps=steps, cfg=cfg,
            )
            scratch_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".wav", prefix="voxcpm-voice-design-",
                    delete=False,
                ) as tmp:
                    scratch_path = tmp.name
                    tmp.write(scratch_audio)
                # Importantly, NO parenthesized instruction or reference
                # transcript is sent for the actual sample. This prevents
                # instruction text from being spoken, and preserves an exact
                # transcript for subsequent high-fidelity voice cloning.
                return self.generate(
                    sample_text, reference_path=scratch_path,
                    seed=seed, steps=steps, cfg=cfg,
                )
            finally:
                if scratch_path is not None:
                    try:
                        os.unlink(scratch_path)
                    except FileNotFoundError:
                        pass

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
                emit("generation.failed", backend="voxcpm2",
                     error_type=type(exc).__name__)
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
                    signal = self._generate_compatible(kwargs, seed)
                except Exception as exc:
                    # VoxCPM2 defaults to torch.compile(..., mode="reduce-overhead").
                    # Recover once from a known compile/graph incompatibility without
                    # disabling acceleration for healthy installations.
                    compiler_error = any(marker.lower() in str(exc).lower()
                                         or marker.lower() in type(exc).__name__.lower()
                                         for marker in COMPILE_ERROR_MARKERS)
                    if not self.optimize or not compiler_error:
                        raise
                    emit("model.switch", backend="voxcpm2",
                         model="voxcpm2", reason="fallback")
                    LOG.warning("[TTS API] Optimized inference incompatible; "
                                "retrying once without torch.compile.")
                    self.optimize = False
                    self.compile_fallback_used = True
                    self._unload_locked()
                    self._load()
                    signal = self._generate_compatible(kwargs, seed)
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
                emit("generation.failed", backend="voxcpm2",
                     error_type=type(exc).__name__)
                raise
            finally:
                self._last_activity = time.monotonic()
                if self.unload_after_request:
                    self._unload_locked()
