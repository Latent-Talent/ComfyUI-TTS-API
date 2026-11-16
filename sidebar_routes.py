"""Same-origin ComfyUI HTTP routes for the TTS API sidebar.

The browser never receives the service's API key or internal control token.
This exposes the same operations to users who can access ComfyUI itself.
"""
from __future__ import annotations

import asyncio
import logging

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData, web

from .managed_service import _MANAGER

LOG = logging.getLogger("comfyui-tts-api")
BASE = "/tts-api"
REQUEST_TIMEOUT = ClientTimeout(total=600, connect=8)


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
                    try:
                        details = await asyncio.to_thread(
                            lambda: raw.decode("utf-8", errors="replace")[:1200]
                        )
                    except Exception:
                        details = "Speech service returned an error"
                    return _error(upstream.status, details)
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
    routes.get(BASE + "/voices")(voices)
    routes.post(BASE + "/voices/design")(design)
    routes.post(BASE + "/voices/clone")(clone)
    routes.delete(BASE + "/voices/{voice_id}")(delete_voice)
    routes.post(BASE + "/speech")(generate)
    routes.post(BASE + "/model/unload")(unload)
    routes.post(BASE + "/server/restart")(restart)
