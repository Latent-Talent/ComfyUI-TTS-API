"""ComfyUI client nodes for the standalone local speech service.

The gateway lives in a separate venv/process; this module needs only ComfyUI's
existing Python packages plus httpx (or requests fallback).
"""
from __future__ import annotations

import io
import os

import torch


def _post(url, payload, key, timeout=600):
    import urllib.request
    import urllib.error
    import json
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS API HTTP {err.code}: {detail[:1000]}") from err


class LocalTTSGenerate:
    CATEGORY = "TTS API"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "statistics")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Hello from ComfyUI TTS."}),
            "voice": ("STRING", {"default": "default"}),
            "model": (["voxcpm2"],),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 2.0, "min": 0, "max": 10, "step": 0.1}),
            "server_url": ("STRING", {"default": "http://127.0.0.1:8025"}),
            "api_key": ("STRING", {"default": ""}),
        }, "optional": {
            "delivery": ("STRING", {"multiline": True, "default": ""}),
        }}

    def generate(self, text, voice, model, seed, steps, cfg, server_url, api_key, delivery=""):
        import soundfile as sf
        raw = _post(server_url.rstrip("/") + "/v1/audio/speech",
                    {"model": model, "input": text, "voice": voice,
                     "response_format": "wav", "instructions": delivery,
                     "seed": seed, "steps": steps, "cfg": cfg}, api_key)
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": sr}, f"{len(audio)/sr:.2f} sec at {sr} Hz")


class LocalTTSClone:
    CATEGORY = "TTS API"
    FUNCTION = "clone"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference_audio": ("AUDIO",),
            "name": ("STRING", {"default": "New character"}),
            "reference_transcript": ("STRING", {"multiline": True, "default": ""}),
            "authorized_to_use": ("BOOLEAN", {"default": False}),
            "server_url": ("STRING", {"default": "http://127.0.0.1:8025"}),
            "api_key": ("STRING", {"default": ""}),
        }}

    def clone(self, reference_audio, name, reference_transcript, authorized_to_use, server_url, api_key):
        import json
        import soundfile as sf
        import urllib.request
        if not authorized_to_use:
            raise ValueError("Confirm that you have authorization to use this voice")
        wave = reference_audio["waveform"]
        sr = reference_audio["sample_rate"]
        if wave.ndim != 3 or wave.shape[0] != 1:
            raise ValueError("Expected a single AUDIO batch")
        buff = io.BytesIO()
        sf.write(buff, wave[0].detach().cpu().T.numpy(), sr, format="WAV")
        boundary = "ComfyTTSBoundary753"
        parts = []
        for name_, value in [("name", name), ("reference_text", reference_transcript),
                             ("consent", "true")]:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name_}\"\r\n\r\n{value}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"reference.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
                     + buff.getvalue() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        req = urllib.request.Request(server_url.rstrip("/") + "/api/voices/clone",
                                     b"".join(parts), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            record = json.loads(response.read())
        return (record["id"],)


class LocalTTSDesign:
    CATEGORY = "TTS API"
    FUNCTION = "design"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "Designed character"}),
            "description": ("STRING", {"multiline": True, "default": "Warm, confident adult voice"}),
            "sample_text": ("STRING", {"multiline": True, "default": "Hello, this is my voice."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            "server_url": ("STRING", {"default": "http://127.0.0.1:8025"}),
            "api_key": ("STRING", {"default": ""}),
        }}

    def design(self, name, description, sample_text, seed, server_url, api_key):
        import json
        raw = _post(server_url.rstrip("/") + "/api/voices/design",
                    {"name": name, "description": description,
                     "sample_text": sample_text, "seed": seed}, api_key)
        return (json.loads(raw)["id"],)


NODE_CLASS_MAPPINGS = {
    "ComfyTTSGenerate": LocalTTSGenerate,
    "ComfyTTSCloneVoice": LocalTTSClone,
    "ComfyTTSDesignVoice": LocalTTSDesign,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyTTSGenerate": "TTS Generate (API)",
    "ComfyTTSCloneVoice": "TTS Clone Voice (API)",
    "ComfyTTSDesignVoice": "TTS Design Voice (API)",
}
