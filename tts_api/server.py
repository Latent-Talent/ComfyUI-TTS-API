"""OpenAI-compatible local TTS API and persistent voice registry."""
from __future__ import annotations

import io
import os
import tempfile
import wave
from pathlib import Path
from typing import Literal

import anyio
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .engine import VoxEngine
from .storage import VoiceStore

ROOT = Path(os.environ.get("TTS_DATA_DIR", "./data")).expanduser().resolve()
STORE = VoiceStore(ROOT / "voices")
MODEL = os.environ.get("TTS_MODEL", "openbmb/VoxCPM2")
ENGINE = VoxEngine(MODEL, os.getenv("TTS_UNLOAD_AFTER_REQUEST", "false").lower() == "true")
KEY = os.getenv("TTS_API_KEY", "")
MAX_TEXT = int(os.getenv("TTS_MAX_TEXT", "12000"))
MAX_UPLOAD = int(os.getenv("TTS_MAX_UPLOAD_MB", "20")) * 1024 * 1024
app = FastAPI(title="ComfyUI TTS API", version="0.1.0")


async def authenticate(authorization: str | None = Header(default=None),
                       x_api_key: str | None = Header(default=None)):
    import hmac
    if KEY and not (hmac.compare_digest(x_api_key or "", KEY) or
                    hmac.compare_digest(authorization or "", f"Bearer {KEY}")):
        raise HTTPException(401, "Invalid or missing API key")


class SpeechRequest(BaseModel):
    model: str = "voxcpm2"
    input: str = Field(min_length=1)
    voice: str = "default"
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str = ""
    seed: int | None = None
    steps: int = Field(default=10, ge=1, le=100)
    cfg: float = Field(default=2.0, ge=0, le=10)


class DesignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    sample_text: str = Field(default="Hello, this is my new voice.", max_length=1000)
    seed: int | None = None


def _render(request: SpeechRequest) -> bytes:
    if len(request.input) > MAX_TEXT:
        raise ValueError(f"Input longer than {MAX_TEXT} characters")
    if request.model not in ("voxcpm2", MODEL):
        raise ValueError("Unknown model. Supported model: voxcpm2")
    if request.speed != 1.0:
        raise ValueError("Speed control is not supported by the native VoxCPM2 backend")
    voice = None if request.voice == "default" else STORE.get(request.voice)
    if request.voice != "default" and voice is None:
        raise KeyError("Unknown voice ID")
    ref = str(STORE.audio_path(voice)) if voice and voice.get("reference_file") else None
    description = (voice or {}).get("description", "") or None
    if request.instructions:
        description = (description + ", " if description else "") + request.instructions
    reference_text = voice.get("reference_text") if voice and ref else None
    return ENGINE.generate(
        request.input, reference_path=ref, reference_text=reference_text or None,
        description=description, seed=request.seed, steps=request.steps, cfg=request.cfg
    )


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL, "loaded": ENGINE.loaded,
            "last_stats": ENGINE.last_stats, "voice_count": len(STORE.list())}


@app.get("/v1/models", dependencies=[Depends(authenticate)])
async def models():
    return {"object": "list", "data": [{"id": "voxcpm2", "object": "model",
                                        "owned_by": "local"}]}


@app.get("/v1/audio/voices", dependencies=[Depends(authenticate)])
async def voices():
    return {"voices": [{"id": "default", "name": "Default"}] +
            [{"id": v["id"], "name": v["name"], "description": v.get("description", "")}
             for v in STORE.list()]}


@app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
async def speech(request: SpeechRequest):
    try:
        audio = await anyio.to_thread.run_sync(_render, request)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, f"Speech generation failed: {exc}") from exc
    if request.response_format == "pcm":
        # OpenAI PCM is raw signed 16-bit little-endian at 24 kHz.
        # Native VoxCPM emits 48 kHz. Fail rather than silently mislabeling.
        raise HTTPException(422, "PCM output is not yet supported; request wav")
    return Response(content=audio, media_type="audio/wav",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/voices/clone", dependencies=[Depends(authenticate)])
async def clone_voice(name: str = Form(...), reference_text: str = Form(""),
                      consent: bool = Form(False), audio: UploadFile = File(...)):
    if not consent:
        raise HTTPException(422, "Voice usage authorization must be confirmed.")
    if not 1 <= len(name) <= 120:
        raise HTTPException(422, "Voice name must contain 1-120 characters")
    if len(reference_text) > 4000:
        raise HTTPException(422, "Reference transcript too long")
    raw = await audio.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Audio exceeds configured upload limit")
    # Decode and re-encode to WAV to avoid trusting extensions or opaque files.
    try:
        signal, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if signal.ndim not in (1, 2) or not 8000 <= sr <= 192000:
            raise ValueError("Unsupported audio")
        if len(signal) / sr > 120 or len(signal) / sr < 0.3:
            raise ValueError("Reference must be between 0.3 and 120 seconds")
        output = io.BytesIO()
        sf.write(output, signal, sr, format="WAV", subtype="PCM_16")
        encoded = output.getvalue()
    except Exception as exc:
        raise HTTPException(422, f"Invalid reference audio: {exc}") from exc
    return STORE.create(name=name, reference=encoded, reference_text=reference_text)


@app.post("/api/voices/design", dependencies=[Depends(authenticate)])
async def design_voice(request: DesignRequest):
    try:
        audio = await anyio.to_thread.run_sync(
            ENGINE.generate, request.sample_text, None, None,
            request.description, request.seed
        )
    except Exception as exc:
        raise HTTPException(503, f"Voice design failed: {exc}") from exc
    return STORE.create(name=request.name, description=request.description,
                        reference=audio, reference_text=request.sample_text)


@app.get("/api/voices", dependencies=[Depends(authenticate)])
async def list_voices():
    return STORE.list()


@app.delete("/api/voices/{voice_id}", dependencies=[Depends(authenticate)])
async def delete_voice(voice_id: str):
    try:
        deleted = STORE.delete(voice_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, "Unknown voice ID")
    return {"deleted": True}


@app.post("/api/model/unload", dependencies=[Depends(authenticate)])
async def unload_model():
    await anyio.to_thread.run_sync(ENGINE.unload)
    return {"loaded": False}
