"""Lazy-loaded VoxCPM2 engine. Requires voxcpm in this service's own venv."""
from __future__ import annotations

import io
import threading
import time
from typing import Optional

import numpy as np
import soundfile as sf


class VoxEngine:
    def __init__(self, model_name: str, unload_after_request: bool = False):
        self.model_name = model_name
        self.unload_after_request = unload_after_request
        self._model = None
        self._lock = threading.RLock()
        self.last_stats: dict = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is None:
            try:
                from voxcpm import VoxCPM
            except ImportError as exc:
                raise RuntimeError("Missing VoxCPM engine. Run: pip install voxcpm") from exc
            self._model = VoxCPM.from_pretrained(self.model_name, load_denoiser=False)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def generate(self, text: str, reference_path: Optional[str] = None,
                 reference_text: Optional[str] = None, description: Optional[str] = None,
                 seed: Optional[int] = None, steps: int = 10, cfg: float = 2.0) -> bytes:
        """Generate a complete WAV in memory; serializes model operations to avoid CUDA races."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        with self._lock:
            self._load()
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
            finally:
                if self.unload_after_request:
                    self.unload()
