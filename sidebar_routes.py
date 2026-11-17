"""Same-origin ComfyUI HTTP routes for the TTS API sidebar.

The browser never receives the service's API key or internal control token.
This exposes the same operations to users who can access ComfyUI itself.
"""
from __future__ import annotations

import asyncio
import json
import hmac
import logging
import os
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData, web

from .managed_service import _MANAGER
from .queue_bridge import BROKER

LOG = logging.getLogger("comfyui-tts-api")
BASE = "/tts-api"
REQUEST_TIMEOUT = ClientTimeout(total=1950, connect=8)


def _headers() -> dict[str, str]:
    key = _MANAGER.config.get("TTS_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _available() -> bool:
    # Only the configured local service, never an arbitrary URL supplied by
    # the frontend. If an independently running server occupies the port,
    # requests are still authenticated against its configured API key.
    return _MANAGER._running() or _MANAGER._port_open()


async def _forward(method: str, path: str, *, payload=None, multipart=None,
                   audio: bool = False) -> web.Response:
    if not _available():
        return _error(503, "The TTS API is not running. Install the service and restart ComfyUI.")
    headers = _headers()
    if payload is not None:
        headers["Content-Type"] = "application/json"
    try:
        async with ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.request(
                method, _MANAGER.url + path, headers=headers,
                json=payload, data=multipart,
            ) as upstream:
                raw = await upstream.read()
                if upstream.status >= 400:
                    details = raw.decode("utf-8", errors="replace")[:1500]
                    try:
                        body = json.loads(details)
                        if isinstance(body, dict):
                            detail = body.get("detail", body.get("error", details))
                            if isinstance(detail, dict):
                                label = detail.get("error", "TTS error")
                                kind = detail.get("exception", "")
                                stage = detail.get("stage", "")
                                message = detail.get("message", "")
                                parts = [str(label)]
                                if kind:
                                    parts.append(f"[{kind}]")
                                if stage:
                                    parts.append(f"({stage})")
                                if message:
                                    parts.append("— " + str(message))
                                details = " ".join(parts)
                            elif isinstance(detail, str):
                                details = detail
                    except (ValueError, TypeError):
                        pass
                    return _error(upstream.status, details[:1200])
                if audio:
                    return web.Response(
                        body=raw,
                        content_type=upstream.headers.get("Content-Type", "audio/wav"),
                        headers={"Cache-Control": "no-store"},
                    )
                return web.Response(
                    body=raw, content_type="application/json",
                    headers={"Cache-Control": "no-store"},
                )
    except (ClientError, asyncio.TimeoutError, OSError) as exc:
        LOG.warning("[TTS API] Sidebar request failed for %s: %s", path, exc)
        return _error(503, "TTS worker unavailable or request timed out")


async def status(_request: web.Request) -> web.Response:
    installed = _MANAGER._python().is_file()
    result = {
        "installed": installed,
        "managed": not _MANAGER._disabled,
        "running": _MANAGER._running(),
        "busy": _MANAGER._queue_busy(),
        "reserved": _MANAGER._reserved,
        "api_url": _MANAGER.url,
        "port": _MANAGER.port,
        "active_model_editable": "TTS_ACTIVE_MODEL" not in os.environ,
        "active_model": _MANAGER.config.get("TTS_ACTIVE_MODEL", "voxcpm2"),
        "api_key": {
            "configured": _MANAGER.api_key_configured,
            "editable": _MANAGER.api_key_editable,
        },
        "config": {
            "idle_unload_seconds": int(_MANAGER.config.get("TTS_IDLE_UNLOAD_SECONDS", 60)),
        },
        "health": None,
    }
    if _available():
        try:
            async with ClientSession(timeout=ClientTimeout(total=1.5)) as session:
                async with session.get(_MANAGER.url + "/health") as upstream:
                    if upstream.status == 200:
                        data = await upstream.json()
                        if data.get("service") == "comfyui-tts-api":
                            result["health"] = data
        except (ClientError, asyncio.TimeoutError, OSError, ValueError):
            pass
    return web.json_response(result, headers={"Cache-Control": "no-store"})


def _verify_queue_token(request: web.Request) -> bool:
    expected = _MANAGER.control_token
    actual = request.headers.get("X-TTS-Control-Token", "")
    return bool(expected) and hmac.compare_digest(actual, expected)


async def queue_acquire(request: web.Request) -> web.Response:
    if not _verify_queue_token(request):
        return _error(403, "Invalid TTS queue control token")
    if not _MANAGER._running():
        return _error(503, "Queue reservations require the managed TTS worker")
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid queue request")
        from server import PromptServer
        from nodes import NODE_CLASS_MAPPINGS
        if "LatentTalentTTSQueueSlot" not in NODE_CLASS_MAPPINGS:
            raise RuntimeError("TTS queue reservation node is not registered")
        prompt_id = BROKER.create(
            data.get("ticket"), PromptServer.instance,
            data.get("kind", "speech"),
        )
    except (ValueError, TypeError) as exc:
        return _error(422, str(exc))
    except (RuntimeError, AttributeError):
        return _error(503, "ComfyUI prompt queue is unavailable")
    return web.json_response({"prompt_id": prompt_id, "state": "pending"})


async def queue_state(request: web.Request) -> web.Response:
    if not _verify_queue_token(request):
        return _error(403, "Invalid TTS queue control token")
    try:
        data = await request.json()
        from server import PromptServer
        state = BROKER.state(data.get("ticket"), PromptServer.instance.prompt_queue)
    except (AttributeError, ValueError, TypeError):
        state = "cancelled"
    return web.json_response({"state": state})


async def queue_release(request: web.Request) -> web.Response:
    if not _verify_queue_token(request):
        return _error(403, "Invalid TTS queue control token")
    try:
        data = await request.json()
        ticket = data.get("ticket")
        if not isinstance(ticket, str):
            raise ValueError("Invalid queue request")
        status = data.get("status", "completed")
        if status not in ("completed", "failed"):
            raise ValueError("Invalid TTS queue result")
        released = BROKER.release(ticket, status=status)
    except (ValueError, TypeError):
        return _error(422, "Invalid queue request")
    return web.json_response({"released": released})


async def models(_request: web.Request) -> web.Response:
    return await _forward("GET", "/api/models")


async def model_defaults(_request: web.Request) -> web.Response:
    return await _forward("GET", "/api/models/defaults")


async def save_model_defaults(request: web.Request) -> web.Response:
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return _error(403, "Model settings must originate from the ComfyUI page")
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, TypeError):
        return _error(400, "Invalid model defaults request")
    return await _forward("POST", "/api/models/defaults", payload=payload)


async def model_download_status(_request: web.Request) -> web.Response:
    return await _forward("GET", "/api/models/download/status")


async def download_model(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
    except (ValueError, TypeError):
        return _error(400, "Invalid download request")
    return await _forward("POST", "/api/models/download", payload=payload)


async def select_model(request: web.Request) -> web.Response:
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return _error(403, "Model settings must originate from the ComfyUI page")
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        result = await asyncio.to_thread(
            _MANAGER.configure_active_model, payload.get("model_id")
        )
    except (ValueError, TypeError) as exc:
        return _error(422, str(exc))
    except PermissionError as exc:
        return _error(403, str(exc))
    except RuntimeError as exc:
        return _error(409, str(exc))
    except OSError:
        LOG.exception("[TTS API] Unable to persist model selection")
        return _error(500, "Unable to persist model selection")
    return web.json_response(result, headers={"Cache-Control": "no-store"})


async def voices(_request: web.Request) -> web.Response:
    return await _forward("GET", "/api/voices")


async def design(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, TypeError):
        return _error(400, "Invalid JSON request")
    return await _forward("POST", "/api/voices/design", payload=payload)


async def clone(request: web.Request) -> web.Response:
    try:
        reader = await request.multipart()
        fields = {}
        wav = None
        max_size = int(_MANAGER.config.get("TTS_MAX_UPLOAD_MB", 20)) * 1024 * 1024
        while part := await reader.next():
            if part.name == "audio":
                buffer = bytearray()
                while chunk := await part.read_chunk(65536):
                    buffer.extend(chunk)
                    if len(buffer) > max_size:
                        return _error(413, "Voice recording exceeds upload limit")
                wav = bytes(buffer)
            elif part.name in ("name", "reference_text", "consent"):
                fields[part.name] = (await part.text())[:4096]
        if not wav:
            return _error(422, "Choose a reference audio file")
        if fields.get("consent") != "true":
            return _error(422, "Voice-use authorization is required")
        form = FormData()
        for key, value in fields.items():
            form.add_field(key, value)
        form.add_field("audio", wav, filename="reference.wav", content_type="audio/wav")
    except (ValueError, RuntimeError, web.HTTPException) as exc:
        return _error(400, str(exc))
    return await _forward("POST", "/api/voices/clone", multipart=form)


def _voice_id_from_route(request: web.Request) -> str | None:
    value = request.match_info.get("voice_id", "")
    if not value.startswith("voice_") or not value.replace("_", "").isalnum():
        return None
    if len(value) > 80:
        return None
    return value


async def voice_details(request: web.Request) -> web.Response:
    voice_id = _voice_id_from_route(request)
    if voice_id is None:
        return _error(422, "Invalid voice ID")
    return await _forward("GET", f"/api/voices/{voice_id}")


async def voice_audio(request: web.Request) -> web.Response:
    voice_id = _voice_id_from_route(request)
    if voice_id is None:
        return _error(422, "Invalid voice ID")
    return await _forward("GET", f"/api/voices/{voice_id}/audio", audio=True)


async def edit_voice(request: web.Request) -> web.Response:
    voice_id = _voice_id_from_route(request)
    if voice_id is None:
        return _error(422, "Invalid voice ID")
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return _error(403, "Voice changes must originate from the ComfyUI page")
    max_size = int(_MANAGER.config.get("TTS_MAX_UPLOAD_MB", 20)) * 1024 * 1024
    fields = {}
    audio = None
    try:
        reader = await request.multipart()
        while part := await reader.next():
            if part.name == "audio":
                data = bytearray()
                while chunk := await part.read_chunk(65536):
                    data.extend(chunk)
                    if len(data) > max_size:
                        return _error(413, "Voice recording exceeds upload limit")
                audio = bytes(data) if data else None
            elif part.name in ("name", "description", "reference_text", "consent"):
                content = await part.read(decode=True)
                if len(content) > 5000:
                    return _error(413, "Voice metadata exceeds size limit")
                fields[part.name] = content.decode("utf-8")
        if not fields.get("name", "").strip():
            return _error(422, "Voice name is required")
        if audio is not None and fields.get("consent") != "true":
            return _error(422, "Voice recording permission must be confirmed")
        form = FormData()
        for key in ("name", "description", "reference_text", "consent"):
            form.add_field(key, fields.get(key, ""))
        if audio is not None:
            form.add_field("audio", audio, filename="reference.bin",
                           content_type="application/octet-stream")
    except (UnicodeError, ValueError, RuntimeError, web.HTTPException):
        return _error(400, "Invalid voice update")
    return await _forward("PATCH", f"/api/voices/{voice_id}", multipart=form)


async def delete_voice(request: web.Request) -> web.Response:
    voice_id = request.match_info["voice_id"]
    if not voice_id.startswith("voice_") or not voice_id.replace("_", "").isalnum():
        return _error(422, "Invalid voice ID")
    return await _forward("DELETE", f"/api/voices/{voice_id}")


async def generate(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, TypeError):
        return _error(400, "Invalid JSON")
    payload["response_format"] = "wav"
    return await _forward("POST", "/v1/audio/speech", payload=payload, audio=True)


async def unload(_request: web.Request) -> web.Response:
    return await _forward("POST", "/api/model/unload")


async def configure_api_key(request: web.Request) -> web.Response:
    """Rotate or set service credentials through the ComfyUI same-origin UI.

    A saved key is never returned by status. Generate returns the key exactly
    once; callers must copy it for SillyTavern before closing the panel.
    """
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return _error(403, "API key updates must originate from the ComfyUI page")
    if request.content_length is not None and request.content_length > 1024:
        return _error(413, "API key request exceeds size limit")
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        action = payload.get("action", "")
        key = payload.get("key")
        result = await asyncio.to_thread(_MANAGER.configure_api_key, action, key)
    except (ValueError, TypeError) as exc:
        return _error(422, str(exc))
    except PermissionError as exc:
        return _error(403, str(exc))
    except RuntimeError as exc:
        return _error(409, str(exc))
    except OSError:
        LOG.exception("[TTS API] Unable to save API key")
        return _error(500, "Unable to persist API key")
    return web.json_response(result, headers={"Cache-Control": "no-store"})


async def restart(_request: web.Request) -> web.Response:
    # Only terminate the child we own; do not touch an independent service.
    with _MANAGER._lock:
        if _MANAGER._disabled:
            return _error(409, "Automatic management is disabled")
        if _MANAGER._queue_busy():
            return _error(409, "Cannot restart the TTS API during a ComfyUI workflow")
        if not _MANAGER._running() and _MANAGER._port_open():
            return _error(409, "Port belongs to an independent service")
        _MANAGER._terminate()
        if not _MANAGER._spawn():
            return _error(503, "Could not start the TTS worker; check the managed-server log")
    return web.json_response({"restarting": True})


def register_routes():
    from server import PromptServer
    routes = PromptServer.instance.routes
    routes.get(BASE + "/status")(status)
    routes.post(BASE + "/queue/acquire")(queue_acquire)
    routes.post(BASE + "/queue/state")(queue_state)
    routes.post(BASE + "/queue/release")(queue_release)
    routes.get(BASE + "/models")(models)
    routes.get(BASE + "/models/defaults")(model_defaults)
    routes.post(BASE + "/models/defaults")(save_model_defaults)
    routes.post(BASE + "/models/download")(download_model)
    routes.get(BASE + "/models/download/status")(model_download_status)
    routes.post(BASE + "/settings/model")(select_model)
    routes.get(BASE + "/voices")(voices)
    routes.post(BASE + "/voices/design")(design)
    routes.post(BASE + "/voices/clone")(clone)
    routes.get(BASE + "/voices/{voice_id}")(voice_details)
    routes.get(BASE + "/voices/{voice_id}/audio")(voice_audio)
    routes.patch(BASE + "/voices/{voice_id}")(edit_voice)
    routes.delete(BASE + "/voices/{voice_id}")(delete_voice)
    routes.post(BASE + "/speech")(generate)
    routes.post(BASE + "/model/unload")(unload)
    routes.post(BASE + "/server/restart")(restart)
    routes.post(BASE + "/settings/api-key")(configure_api_key)
