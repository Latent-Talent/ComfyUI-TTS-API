"""OpenAI-compatible local TTS API and persistent voice registry."""
from __future__ import annotations

import asyncio
import hmac
import io
import logging
import os
from contextlib import asynccontextmanager
import subprocess
import shutil
from pathlib import Path
from typing import Literal

import anyio
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .engine import VoxEngine
from .storage import VoiceStore

LOG = logging.getLogger("comfyui-tts-api.server")

ROOT = Path(os.environ.get("TTS_DATA_DIR", "./data")).expanduser().resolve()
STORE = VoiceStore(ROOT / "voices")
MODEL = os.environ.get("TTS_MODEL", "openbmb/VoxCPM2")
ENGINE = VoxEngine(MODEL, os.getenv("TTS_UNLOAD_AFTER_REQUEST", "false").lower() == "true")
KEY = os.getenv("TTS_API_KEY", "")
MAX_TEXT = int(os.getenv("TTS_MAX_TEXT", "12000"))
MAX_UPLOAD = int(os.getenv("TTS_MAX_UPLOAD_MB", "20")) * 1024 * 1024
CONTROL_TOKEN = os.getenv("TTS_CONTROL_TOKEN", "")
IDLE_SECONDS = max(0, int(os.getenv("TTS_IDLE_UNLOAD_SECONDS", "60")))
PARENT_PID = int(os.getenv("TTS_PARENT_PID", "0"))


async def _reap_idle_model():
    while True:
        await asyncio.sleep(5)
        # A managed child must not persist after its ComfyUI parent crashes.
        if PARENT_PID and os.getppid() != PARENT_PID:
            os._exit(0)
        if IDLE_SECONDS:
            await anyio.to_thread.run_sync(ENGINE.unload_if_idle, IDLE_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_reap_idle_model())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await anyio.to_thread.run_sync(ENGINE.unload)


app = FastAPI(title="ComfyUI TTS API", version="0.2.0", lifespan=lifespan)


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
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "mp3"
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
    voice = None if request.voice == "default" else STORE.get(request.voice)
    if request.voice != "default" and voice is None:
        raise KeyError("Unknown voice ID")
    ref = str(STORE.audio_path(voice)) if voice and voice.get("reference_file") else None
    # VoxCPM2 takes voice/style instructions by prepending "(description)"
    # to text, but some installations speak that control text aloud.
    # Playback must *only* synthesize request.input. For saved voices, the
    # stored audio reference already captures the designed timbre.
    # The 'instructions' field is intentionally not injected until a backend
    # offers a separate, non-spoken conditioning channel.
    reference_text = voice.get("reference_text") if voice and ref else None
    return ENGINE.generate(
        request.input, reference_path=ref, reference_text=reference_text or None,
        seed=request.seed, steps=request.steps, cfg=request.cfg
    )


@app.get("/health")
async def health():
    return {"ok": True, "service": "comfyui-tts-api", "model": MODEL,
            "cuda_allocator": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "backend:native")),
            "loaded": ENGINE.loaded, "paused": ENGINE.paused,
            "compile_enabled": ENGINE.optimize,
            "compile_fallback_used": ENGINE.compile_fallback_used,
            "last_error": ENGINE.last_error,
            "idle_unload_seconds": IDLE_SECONDS, "last_stats": ENGINE.last_stats,
            "voice_count": len(STORE.list())}


def _authorize_control(value: str | None):
    if not CONTROL_TOKEN or not hmac.compare_digest(value or "", CONTROL_TOKEN):
        raise HTTPException(403, "Control operations require an internal token")


@app.post("/api/internal/pause", include_in_schema=False)
async def internal_pause(x_tts_control_token: str | None = Header(default=None)):
    _authorize_control(x_tts_control_token)
    if not await anyio.to_thread.run_sync(ENGINE.try_pause):
        raise HTTPException(409, "TTS is currently generating audio")
    return {"paused": True, "loaded": ENGINE.loaded}


@app.post("/api/internal/resume", include_in_schema=False)
async def internal_resume(x_tts_control_token: str | None = Header(default=None)):
    _authorize_control(x_tts_control_token)
    await anyio.to_thread.run_sync(ENGINE.resume)
    return {"paused": False, "loaded": ENGINE.loaded}


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
    except Exception as exc:
        # Third-party TTS backends may raise AssertionError, TypeError,
        # ValueError, CUDA exceptions, or custom compiler exceptions. Do not
        # turn them into an opaque HTTP 500 that says only "Internal Server Error".
        LOG.exception("[TTS API] Unhandled speech error")
        if ENGINE.last_error is None:
            ENGINE.last_error = {"stage": "speech_api", "type": type(exc).__name__,
                                 "message": str(exc)[:1000]}
        message = str(exc).strip()[:1000] or "No error message supplied"
        raise HTTPException(503, detail={
            "error": "Speech generation failed",
            "exception": type(exc).__name__,
            "message": message,
            "stage": ENGINE.last_error.get("stage", "unknown"),
            "compile_enabled": ENGINE.optimize,
        }) from exc
    if request.response_format == "wav" and request.speed == 1.0:
        return Response(content=audio, media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})
    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "FFmpeg is required for this output format or speed")
    codec = {"mp3": ("libmp3lame", "mp3", "audio/mpeg"),
             "opus": ("libopus", "ogg", "audio/ogg"),
             "aac": ("aac", "adts", "audio/aac"),
             "flac": ("flac", "flac", "audio/flac"),
             "pcm": ("pcm_s16le", "s16le", "application/octet-stream"),
             "wav": ("pcm_s16le", "wav", "audio/wav")}
    encoder, container, mime = codec[request.response_format]
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", "pipe:0", "-vn"]
    if request.speed != 1.0:
        rate = request.speed
        factors = []
        while rate > 2:
            factors.append(2.0)
            rate /= 2
        while rate < 0.5:
            factors.append(0.5)
            rate /= 0.5
        factors.append(rate)
        args += ["-af", ",".join(f"atempo={factor:.5f}" for factor in factors)]
    if request.response_format == "pcm":
        args += ["-ar", "24000", "-ac", "1"]
    args += ["-codec:a", encoder, "-f", container, "pipe:1"]
    result = await anyio.to_thread.run_sync(
        lambda: subprocess.run(args, input=audio, capture_output=True, timeout=120))
    if result.returncode != 0:
        raise HTTPException(503, "Audio conversion failed: " +
                            result.stderr.decode("utf-8", errors="replace")[:300])
    return Response(content=result.stdout, media_type=mime,
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
            ENGINE.design_voice, request.sample_text,
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
