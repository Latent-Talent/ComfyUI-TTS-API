"""Optional Qwen3-TTS engine using the official qwen-tts Python package.

Only one Qwen checkpoint is loaded at once. Imports/weights remain lazy so
existing VoxCPM installations continue to run without qwen-tts installed.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import gc
import io
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import QWEN_MODELS, QWEN_SPEAKERS_LOOKUP

LOG = logging.getLogger("comfyui-tts-api.qwen")


class QwenEngine:
    def __init__(self, unload_after_request: bool = False):
        self._model = None
        self._model_id: str | None = None
        self._lock = threading.RLock()
        self._paused = False
        self._last_activity = time.monotonic()
        self.unload_after_request = unload_after_request
        self.last_stats: dict = {}
        self.last_error: dict | None = None
        self._prompt_cache: OrderedDict[tuple, object] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model(self) -> str | None:
        return self._model_id if self.loaded else None

    @property
    def paused(self) -> bool:
        return self._paused

    def _unload_locked(self) -> None:
        self._prompt_cache.clear()
        model = self._model
        self._model = None
        self._model_id = None
        del model
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._last_activity = time.monotonic()

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

    def _load(self, model_id: str) -> None:
        if self.loaded and self._model_id == model_id:
            return
        if model_id not in QWEN_MODELS:
            raise ValueError(f"Unknown Qwen-TTS checkpoint: {model_id}")
        if self.loaded:
            self._unload_locked()
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen-TTS is not installed in the TTS worker. Run "
                "'bash install-qwen.sh' in ComfyUI-TTS-API, then restart "
                "the managed TTS API from its sidebar."
            ) from exc
        is_cuda = torch.cuda.is_available()
        kwargs = {
            "device_map": "cuda:0" if is_cuda else "cpu",
            "dtype": torch.bfloat16 if is_cuda else torch.float32,
            "attn_implementation": "sdpa" if is_cuda else "eager",
        }
        repo = QWEN_MODELS[model_id]["repo"]
        LOG.info("[TTS API] Loading %s (downloads weights on first use)", repo)
        self._model = Qwen3TTSModel.from_pretrained(repo, **kwargs)
        self._model_id = model_id
        LOG.info("[TTS API] Loaded %s", repo)

    @staticmethod
    def _encode(wavs, sample_rate: int) -> bytes:
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio")
        signal = np.asarray(wavs[0], dtype=np.float32).squeeze()
        if signal.ndim != 1 or not signal.size:
            raise RuntimeError("Qwen3-TTS returned invalid audio")
        stream = io.BytesIO()
        sf.write(stream, signal, int(sample_rate), format="WAV", subtype="PCM_16")
        return stream.getvalue()

    def generate(
        self, model_id: str, text: str, *,
        voice: str = "default", reference_path: str | None = None,
        reference_text: str | None = None,
        instructions: str = "", language: str = "Auto",
        seed: int | None = None,
    ) -> bytes:
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        if model_id not in QWEN_MODELS:
            raise ValueError(f"Unknown Qwen3-TTS model {model_id!r}")
        with self._lock:
            if self._paused:
                raise RuntimeError("Qwen3-TTS paused while ComfyUI needs GPU memory")
            self.last_error = None
            start = time.perf_counter()
            try:
                self._load(model_id)
                self._last_activity = time.monotonic()
                # Qwen's public generation methods do not specify a seed
                # argument. Seed torch inside this isolated service thread
                # rather than forwarding an unknown kwarg to Transformers.
                torch = sys.modules.get("torch")
                context = (
                    torch.random.fork_rng(
                        devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
                    ) if seed is not None and torch is not None else nullcontext()
                )
                with context:
                    if seed is not None and torch is not None:
                        torch.manual_seed(int(seed))
                    kind = QWEN_MODELS[model_id]["kind"]
                    if kind == "customvoice":
                        speaker = QWEN_SPEAKERS_LOOKUP.get(voice.strip().casefold())
                        if speaker is None:
                            speaker = "Ryan" if voice.strip().casefold() in (
                                "", "default", "alloy", "echo", "nova",
                                "fable", "onyx", "shimmer", "ash", "ballad",
                                "coral", "sage", "verse", "marin", "cedar"
                            ) else None
                        if speaker is None:
                            raise ValueError(
                                f"Unknown Qwen speaker: {voice!r}. Choose "
                                "one of " + ", ".join(QWEN_SPEAKERS_LOOKUP.values())
                            )
                        wavs, sr = self._model.generate_custom_voice(
                            text=text, speaker=speaker, language=language,
                            instruct=instructions or "",
                        )
                    elif kind == "voicedesign":
                        wavs, sr = self._model.generate_voice_design(
                            text=text, instruct=instructions or "A natural, clear voice.",
                            language=language,
                        )
                    else:
                        if not reference_path:
                            raise ValueError(
                                "Qwen Base requires a saved voice recording. "
                                "Use CustomVoice for preset speakers or select a saved voice."
                            )
                        path = Path(reference_path)
                        st = path.stat()
                        cache_key = (
                            model_id, str(path), st.st_mtime_ns, st.st_size,
                            reference_text or "",
                        )
                        prompt = self._prompt_cache.get(cache_key)
                        if prompt is None:
                            prompt = self._model.create_voice_clone_prompt(
                                ref_audio=str(path),
                                ref_text=reference_text or None,
                                x_vector_only_mode=not bool(reference_text),
                            )
                            self._prompt_cache[cache_key] = prompt
                            while len(self._prompt_cache) > 8:
                                self._prompt_cache.popitem(last=False)
                        else:
                            self._prompt_cache.move_to_end(cache_key)
                        wavs, sr = self._model.generate_voice_clone(
                            text=text, language=language,
                            voice_clone_prompt=prompt,
                            non_streaming_mode=True,
                        )
                audio = self._encode(wavs, sr)
                elapsed = time.perf_counter() - start
                duration = len(np.asarray(wavs[0]).squeeze()) / int(sr)
                self.last_stats = {
                    "seconds": round(elapsed, 3),
                    "audio_seconds": round(duration, 3),
                    "real_time_factor": round(elapsed / duration, 3) if duration else None,
                    "model": model_id,
                }
                return audio
            except Exception as exc:
                self.last_error = {"stage": "qwen_generation", "type": type(exc).__name__,
                                   "message": str(exc)[:1000]}
                LOG.exception("[TTS API] Qwen3-TTS generation failed")
                raise
            finally:
                self._last_activity = time.monotonic()
                if self.unload_after_request:
                    self._unload_locked()
