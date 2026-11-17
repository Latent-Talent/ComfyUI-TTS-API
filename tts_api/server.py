"""OpenAI-compatible local TTS API and persistent voice registry."""
from __future__ import annotations

import asyncio
import hmac
import io
import importlib.util
import logging
import os
import threading
import time
import uuid
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
from .qwen_engine import QwenEngine
from .comfy_queue import queued_tts
from .operations import REQUEST_ID, configure as configure_operations, emit
from .models import (
    VOX_ID, QWEN_DEFAULT, QWEN_DESIGN, QWEN_MODELS, QWEN_SPEAKERS,
    QWEN_SPEAKERS_LOOKUP, OPENAI_MODEL_ALIASES, OPENAI_VOICE_ALIASES,
    canonical_model, is_qwen, qwen_base_for, model_catalog,
)

LOG = logging.getLogger("comfyui-tts-api.server")

ROOT = Path(os.environ.get("TTS_DATA_DIR", "./data")).expanduser().resolve()
STORE = VoiceStore(ROOT / "voices")
MODEL = os.environ.get("TTS_MODEL", "openbmb/VoxCPM2")
ENGINE = VoxEngine(MODEL, os.getenv("TTS_UNLOAD_AFTER_REQUEST", "false").lower() == "true")
QWEN_ENGINE = QwenEngine(
    os.getenv("TTS_UNLOAD_AFTER_REQUEST", "false").lower() == "true"
)
ACTIVE_MODEL = canonical_model(
    os.getenv("TTS_ACTIVE_MODEL", VOX_ID), default=VOX_ID,
    voxcpm_checkpoint=MODEL,
)
INFERENCE_LOCK = threading.RLock()

DOWNLOAD_LOCK = threading.RLock()
DOWNLOAD_STATUS = {"state": "idle", "model_id": None, "error": None}

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
            await anyio.to_thread.run_sync(QWEN_ENGINE.unload_if_idle, IDLE_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_operations(ROOT)
    emit("service.started", model=ACTIVE_MODEL, status="running")
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
        await anyio.to_thread.run_sync(QWEN_ENGINE.unload)
        emit("service.stopped", reason="shutdown")


app = FastAPI(title="ComfyUI TTS API", version="0.5.0", lifespan=lifespan)

@app.middleware("http")
async def operation_request_logging(request, call_next):
    kind = {
        "/v1/audio/speech": "speech",
        "/api/voices/design": "voice_design",
        "/api/voices/clone": "voice_clone",
        "/api/models/download": "download",
    }.get(request.url.path)
    if kind is None or request.method != "POST":
        return await call_next(request)
    token = REQUEST_ID.set(uuid.uuid4().hex[:12])
    start = time.perf_counter()
    emit("request.started", kind=kind, status="starting")
    try:
        response = await call_next(request)
        emit(
            "request.completed" if response.status_code < 400 else "request.failed",
            kind=kind, status="ok" if response.status_code < 400 else "error",
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        return response
    except Exception as exc:
        emit("request.failed", kind=kind, status="error",
             error_type=type(exc).__name__,
             elapsed_ms=(time.perf_counter() - start) * 1000)
        raise
    finally:
        REQUEST_ID.reset(token)



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
    instructions: str = Field(default="", max_length=1000)
    seed: int | None = None
    steps: int = Field(default=10, ge=1, le=100)
    cfg: float = Field(default=2.0, ge=0, le=10)
    language: str = Field(default="Auto", max_length=40)


class DesignRequest(BaseModel):
    model: str | None = None
    language: str = Field(default="Auto", max_length=40)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    sample_text: str = Field(default="Hello, this is my new voice.", max_length=1000)
    seed: int | None = None


def _resolve_voice(value: str, *, backend: str = VOX_ID) -> dict | None:
    """Resolve a SillyTavern voice by saved ID, unique name, or built-in alias.

    SillyTavern treats comma-separated voice entries as literal strings, so
    entries such as " Nova " need whitespace and case normalization. Never
    silently fall back to default for an unknown character.
    """
    requested = value.strip()
    if not requested:
        raise HTTPException(
            status_code=404, detail="Empty voice name. Use 'default' or a saved voice ID/name."
        )

    saved = STORE.list()
    # First preserve exact persistent ID semantics. Name matching is only
    # performed when there was no exact saved-ID match.
    for record in saved:
        if record.get("id") == requested:
            return record

    matches = [
        record for record in saved
        if isinstance(record.get("name"), str)
        and record["name"].strip().casefold() == requested.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        identifiers = ", ".join(record["id"] for record in matches)
        raise HTTPException(
            status_code=422,
            detail=f"Voice name {requested!r} is ambiguous. Use a saved voice ID: {identifiers}",
        )

    if requested.casefold() == "default" or requested.casefold() in OPENAI_VOICE_ALIASES:
        return None
    if is_qwen(backend) and requested.casefold() in QWEN_SPEAKERS_LOOKUP:
        return None

    available = ["default"] + [record["id"] for record in saved]
    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown voice {requested!r}. Use a saved voice ID/name from "
            "/v1/audio/voices, or an OpenAI default voice such as 'alloy'. "
            f"Available saved IDs: {', '.join(available[:12])}"
        ),
    )


def _render(request: SpeechRequest) -> bytes:
    if len(request.input) > MAX_TEXT:
        raise ValueError(f"Input longer than {MAX_TEXT} characters")
    model_id = canonical_model(request.model, ACTIVE_MODEL, MODEL)
    voice = _resolve_voice(request.voice, backend=model_id)
    ref = str(STORE.audio_path(voice)) if voice and voice.get("reference_file") else None
    reference_text = voice.get("reference_text") if voice and ref else None
    emit("generation.started", model=model_id,
         backend="qwen3-tts" if is_qwen(model_id) else "voxcpm2",
         voice_type="saved" if voice is not None
         else ("preset" if is_qwen(model_id) else "default"),
         has_reference=bool(ref), has_instructions=bool(request.instructions),
         characters=len(request.input), steps=request.steps if not is_qwen(model_id) else 0)

    # All backend switching occurs inside this serialized section. No two GPU
    # models are retained across VoxCPM / Qwen3-TTS requests.
    with INFERENCE_LOCK:
        if is_qwen(model_id):
            if ENGINE.loaded:
                emit("model.switch", backend="qwen3-tts",
                     model=model_id, reason="switch")
                ENGINE.unload()
            actual_model = model_id
            if voice is not None:
                if not ref:
                    raise ValueError("Saved voice has no reference audio to clone")
                actual_model = qwen_base_for(model_id)
            result = QWEN_ENGINE.generate(
                actual_model, request.input, voice=request.voice,
                reference_path=ref, reference_text=reference_text,
                instructions=request.instructions, language=request.language,
                seed=request.seed,
            )
            emit("generation.completed", model=actual_model, backend="qwen3-tts",
                 audio_bytes=len(result), audio_seconds=QWEN_ENGINE.last_stats.get("audio_seconds", 0.0),
                 elapsed_ms=QWEN_ENGINE.last_stats.get("seconds", 0.0) * 1000)
            return result
        if QWEN_ENGINE.loaded:
            emit("model.switch", backend="voxcpm2",
                 model=VOX_ID, reason="switch")
            QWEN_ENGINE.unload()
        result = ENGINE.generate_with_delivery(
            request.input, instructions=request.instructions,
            reference_path=ref, reference_text=reference_text or None,
            seed=request.seed, steps=request.steps, cfg=request.cfg,
        )
        emit("generation.completed", model=VOX_ID, backend="voxcpm2",
             audio_bytes=len(result), audio_seconds=ENGINE.last_stats.get("audio_seconds", 0.0),
             elapsed_ms=ENGINE.last_stats.get("seconds", 0.0) * 1000)
        return result


@app.get("/health")
async def health():
    qwen_running = QWEN_ENGINE.loaded
    return {
        "ok": True, "service": "comfyui-tts-api", "version": app.version,
        "model": ACTIVE_MODEL, "active_model": ACTIVE_MODEL,
        "loaded_model": QWEN_ENGINE.loaded_model if qwen_running else (VOX_ID if ENGINE.loaded else None),
        "voxcpm_checkpoint": MODEL,
        "qwen_installed": importlib.util.find_spec("qwen_tts") is not None,
        "qwen_loaded": qwen_running,
        "cuda_allocator": os.environ.get(
            "PYTORCH_ALLOC_CONF",
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "backend:native"),
        ),
        "loaded": ENGINE.loaded or qwen_running,
        "paused": ENGINE.paused or QWEN_ENGINE.paused,
        "compile_enabled": ENGINE.optimize,
        "compile_fallback_used": ENGINE.compile_fallback_used,
        "last_error": QWEN_ENGINE.last_error if qwen_running else ENGINE.last_error,
        "idle_unload_seconds": IDLE_SECONDS,
        "last_stats": QWEN_ENGINE.last_stats if qwen_running else ENGINE.last_stats,
        "voice_count": len(STORE.list()),
    }


def _authorize_control(value: str | None):
    if not CONTROL_TOKEN or not hmac.compare_digest(value or "", CONTROL_TOKEN):
        raise HTTPException(403, "Control operations require an internal token")


def _pause_both() -> bool:
    vox_ok = ENGINE.try_pause()
    qwen_ok = QWEN_ENGINE.try_pause()
    return vox_ok and qwen_ok


def _resume_both():
    ENGINE.resume()
    QWEN_ENGINE.resume()


def _unload_both():
    with INFERENCE_LOCK:
        ENGINE.unload()
        QWEN_ENGINE.unload()


@app.post("/api/internal/pause", include_in_schema=False)
async def internal_pause(x_tts_control_token: str | None = Header(default=None)):
    _authorize_control(x_tts_control_token)
    if not await anyio.to_thread.run_sync(_pause_both):
        emit("worker.pause_blocked", reason="workflow", status="error")
        raise HTTPException(409, "TTS is currently generating audio")
    emit("worker.paused", reason="workflow", status="paused")
    return {"paused": True, "loaded": ENGINE.loaded or QWEN_ENGINE.loaded}


@app.post("/api/internal/resume", include_in_schema=False)
async def internal_resume(x_tts_control_token: str | None = Header(default=None)):
    _authorize_control(x_tts_control_token)
    await anyio.to_thread.run_sync(_resume_both)
    emit("worker.resumed", reason="workflow", status="running")
    return {"paused": False, "loaded": ENGINE.loaded or QWEN_ENGINE.loaded}


@app.get("/v1/models", dependencies=[Depends(authenticate)])
async def models():
    return {"object": "list", "data": [
        {"id": VOX_ID, "object": "model", "owned_by": "local"},
        *[{"id": name, "object": "model", "owned_by": "local-qwen3-tts"}
          for name in QWEN_MODELS],
        *[{"id": alias, "object": "model",
           "owned_by": "local-active-backend-alias"}
          for alias in sorted(OPENAI_MODEL_ALIASES)],
    ]}


@app.get("/v1/audio/voices", dependencies=[Depends(authenticate)])
async def voices():
    return {"voices": [
        {"id": "default", "name": "Default"},
        *[{"id": name, "name": name, "source": "qwen-customvoice"}
          for name in QWEN_SPEAKERS],
        *[{"id": v["id"], "name": v["name"],
           "description": v.get("description", ""), "source": "saved"}
          for v in STORE.list()],
    ]}


@app.get("/api/models", dependencies=[Depends(authenticate)])
async def list_models():
    models = model_catalog()
    try:
        from huggingface_hub import try_to_load_from_cache
        for record in models:
            if "repo" in record:
                cached = try_to_load_from_cache(record["repo"], "config.json")
                record["cache_detected"] = isinstance(cached, str)
    except ImportError:
        pass
    return {
        "active_model": ACTIVE_MODEL,
        "models": models,
        "qwen_installed": importlib.util.find_spec("qwen_tts") is not None,
        "loaded_model": QWEN_ENGINE.loaded_model if QWEN_ENGINE.loaded
                        else (VOX_ID if ENGINE.loaded else None),
    }


class ModelDownloadRequest(BaseModel):
    model_id: str


def _download_model(model_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(QWEN_MODELS[model_id]["repo"])
    except Exception as exc:
        emit("download.failed", model=model_id, error_type=type(exc).__name__)
        with DOWNLOAD_LOCK:
            DOWNLOAD_STATUS.update({
                "state": "failed", "model_id": model_id,
                "error": str(exc)[:500],
            })
    else:
        emit("download.completed", model=model_id)
        with DOWNLOAD_LOCK:
            DOWNLOAD_STATUS.update({
                "state": "complete", "model_id": model_id, "error": None,
            })


@app.post("/api/models/download", dependencies=[Depends(authenticate)])
async def download_model(request: ModelDownloadRequest):
    if request.model_id not in QWEN_MODELS:
        raise HTTPException(422, "Select one of the supported Qwen model IDs")
    with DOWNLOAD_LOCK:
        if DOWNLOAD_STATUS["state"] == "downloading":
            raise HTTPException(
                409,
                f"Downloading {DOWNLOAD_STATUS['model_id']} already",
            )
        DOWNLOAD_STATUS.update({
            "state": "downloading", "model_id": request.model_id,
            "error": None,
        })
        emit("download.started", model=request.model_id)
        threading.Thread(
            target=_download_model, args=(request.model_id,),
            name="TTS-Qwen-HF-Download", daemon=True,
        ).start()
        return dict(DOWNLOAD_STATUS)


@app.get("/api/models/download/status", dependencies=[Depends(authenticate)])
async def model_download_status():
    with DOWNLOAD_LOCK:
        return dict(DOWNLOAD_STATUS)


@app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
async def speech(request: SpeechRequest):
    # Actual ComfyUI output-node job occupies a queue position until the
    # complete audio response is prepared, including FFmpeg conversion.
    async with queued_tts("speech"):
        return await _speech_in_queue(request)


async def _speech_in_queue(request: SpeechRequest):
    try:
        audio = await anyio.to_thread.run_sync(_render, request)
    except HTTPException:
        # Preserve 404/422 voice lookup details rather than wrapping them in
        # an unrelated 503 generation error.
        raise
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        # Do not log third-party exception messages or tracebacks: they can
        # contain user input in their argument values.
        emit("generation.failed", error_type=type(exc).__name__, kind="speech")
        resolved = canonical_model(request.model, ACTIVE_MODEL, MODEL)
        engine = QWEN_ENGINE if is_qwen(resolved) else ENGINE
        if engine.last_error is None:
            engine.last_error = {"stage": "speech_api",
                                 "type": type(exc).__name__, "message": str(exc)[:1000]}
        message = str(exc).strip()[:1000] or "No error message supplied"
        raise HTTPException(503, detail={
            "error": "Speech generation failed",
            "exception": type(exc).__name__,
            "message": message,
            "stage": engine.last_error.get("stage", "unknown"),
            "model": resolved,
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
    record = STORE.create(name=name, reference=encoded, reference_text=reference_text)
    emit("voice.clone.saved", kind="voice_clone", audio_bytes=len(encoded))
    return record


def _design_voice(request: DesignRequest):
    selected = canonical_model(request.model, ACTIVE_MODEL, MODEL)
    emit("voice.design.started", model=QWEN_DESIGN if is_qwen(selected) else VOX_ID,
         backend="qwen3-tts" if is_qwen(selected) else "voxcpm2",
         characters=len(request.sample_text), has_instructions=True)
    with INFERENCE_LOCK:
        if is_qwen(selected):
            # VoiceDesign is a dedicated checkpoint. The resulting reference
            # is stored as an ordinary persistent voice and cloned by Base.
            if ENGINE.loaded:
                ENGINE.unload()
            return QWEN_ENGINE.generate(
                QWEN_DESIGN, request.sample_text, instructions=request.description,
                language=request.language, seed=request.seed,
            )
        if QWEN_ENGINE.loaded:
            QWEN_ENGINE.unload()
        return ENGINE.design_voice(
            request.sample_text, request.description, request.seed
        )


@app.post("/api/voices/design", dependencies=[Depends(authenticate)])
async def design_voice(request: DesignRequest):
    async with queued_tts("voice_design"):
        return await _design_voice_in_queue(request)


async def _design_voice_in_queue(request: DesignRequest):
    try:
        audio = await anyio.to_thread.run_sync(_design_voice, request)
    except Exception as exc:
        emit("voice.design.failed", error_type=type(exc).__name__, kind="voice_design")
        raise HTTPException(503, f"Voice design failed: {exc}") from exc
    record = STORE.create(
        name=request.name, description=request.description,
        reference=audio, reference_text=request.sample_text,
    )
    emit("voice.design.completed", kind="voice_design",
         characters=len(request.sample_text), audio_bytes=len(audio))
    return record


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
    emit("voice.deleted")
    return {"deleted": True}


@app.post("/api/model/unload", dependencies=[Depends(authenticate)])
async def unload_model():
    await anyio.to_thread.run_sync(_unload_both)
    emit("model.unloaded", reason="manual")
    return {"loaded": False}
