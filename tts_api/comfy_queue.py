"""Worker side of the real ComfyUI queue reservation protocol.

No speech text, voice descriptions, audio, or credentials are sent as queue
inputs. Only an opaque random ticket and operation kind cross the boundary.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid

import anyio

QUEUE_WAIT_SECONDS = 1800


class QueueUnavailable(RuntimeError):
    pass


# A GPU/Auto request gets a real ticket only once the ComfyUI broker starts
# its reservation. Explicit CPU requests use a distinct (None, kind) scope
# which permits CPU-only loads but cannot authorize GPU operations. ContextVars
# propagate to anyio worker threads, including model load inside _render.
# Opaque tickets are never logged, returned in errors, or stored in models.
_ACTIVE_RESERVATION: ContextVar[tuple[str | None, str] | None] = ContextVar(
    "tts_comfy_active_reservation", default=None
)


def active_request_kind() -> str | None:
    reservation = _ACTIVE_RESERVATION.get()
    return reservation[1] if reservation else None


def require_queue_reservation(device: str = "auto") -> None:
    """Authorize GPU via queue ticket or explicitly CPU via CPU request scope.

    A CPU bypass marker is not a general-purpose queue ticket: even a cached
    GPU model or an Auto device request must still hold a genuine reservation.
    """
    managed = bool(
        os.getenv("TTS_COMFY_QUEUE_URL") or os.getenv("TTS_CONTROL_TOKEN")
    )
    if managed:
        reservation = _ACTIVE_RESERVATION.get()
        if reservation is None or (
            not reservation[0] and not (reservation[1] in ("speech", "voice_design")
                                        and device == "cpu")
        ):
            raise QueueUnavailable(
                "TTS model loading requires an active ComfyUI queue reservation "
                "unless the request explicitly uses CPU"
            )


class QueueClient:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url if url is not None else os.getenv("TTS_COMFY_QUEUE_URL", "")).rstrip("/")
        self.token = token if token is not None else os.getenv("TTS_CONTROL_TOKEN", "")

    def _post(self, endpoint: str, ticket: str, kind: str = "speech",
              status: str | None = None) -> dict:
        if not self.url or not self.token:
            raise QueueUnavailable("Managed TTS queue bridge is unconfigured")
        payload = {"ticket": ticket, "kind": kind}
        if endpoint == "release" and status is not None:
            payload["status"] = status
        data = json.dumps(payload).encode("ascii")
        request = urllib.request.Request(
            self.url + "/" + endpoint, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "X-TTS-Control-Token": self.token},
        )
        try:
            # The only permitted TLS bypass is for the localhost control
            # bridge, where both processes are owned by this ComfyUI instance
            # and requests are authenticated by an unguessable bearer token.
            context = None
            if self.url.startswith("https://127.0.0.1:"):
                context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=5.0, context=context) as response:
                return json.loads(response.read(4096))
        except (urllib.error.HTTPError, urllib.error.URLError,
                OSError, TimeoutError, ValueError) as exc:
            # Do not leak request contents, server exception bodies, or tokens.
            raise QueueUnavailable(
                f"Unable to {endpoint} ComfyUI TTS queue slot ({type(exc).__name__})"
            ) from None

    def acquire(self, kind: str) -> str | None:
        # Standalone (bash start.sh) is allowed to operate without ComfyUI.
        # A managed worker is NEVER allowed to bypass a broken queue bridge.
        if not self.url and not self.token:
            return None
        if not self.url or not self.token:
            raise QueueUnavailable("TTS queue bridge configuration is incomplete")
        ticket = uuid.uuid4().hex
        self._post("acquire", ticket, kind=kind)
        start = time.monotonic()
        try:
            while time.monotonic() - start < QUEUE_WAIT_SECONDS:
                state = self._post("state", ticket).get("state")
                if state == "running":
                    return ticket
                if state in ("cancelled", "completed"):
                    raise QueueUnavailable("The ComfyUI TTS queue reservation was cancelled")
                if state not in ("pending", "activating"):
                    raise QueueUnavailable("Unknown TTS queue state")
                time.sleep(0.25)
            raise QueueUnavailable("Timed out waiting for the ComfyUI TTS queue slot")
        except BaseException:
            try:
                self._post("release", ticket)
            except QueueUnavailable:
                pass
            raise

    def release(self, ticket: str | None, status: str = "completed"):
        if ticket:
            self._post("release", ticket, status=status)


@asynccontextmanager
async def queued_tts(kind: str):
    client = QueueClient()
    ticket = await anyio.to_thread.run_sync(client.acquire, kind)
    # Mark this request as authorized only after successful acquisition.
    # This context is inherited by the thread doing inference.
    reservation_token = _ACTIVE_RESERVATION.set((ticket, kind))
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        try:
            if ticket:
                try:
                    # Successful API responses get a TTS completion marker.
                    # Failed/cancelled requests remain failed in ComfyUI history.
                    await anyio.to_thread.run_sync(
                        client.release, ticket,
                        "completed" if succeeded else "failed",
                    )
                except QueueUnavailable:
                    pass
        finally:
            _ACTIVE_RESERVATION.reset(reservation_token)



@asynccontextmanager
async def cpu_tts(kind: str):
    """Request-scoped CPU-only authorization; never contacts ComfyUI's queue.

    Used only for the explicit CPU runtime choice. A backend attempting to
    load or generate with Auto/CUDA inside this scope fails closed.
    """
    if kind not in ("speech", "voice_design"):
        raise ValueError("Unsupported CPU TTS operation")
    token = _ACTIVE_RESERVATION.set((None, kind))
    try:
        yield
    finally:
        _ACTIVE_RESERVATION.reset(token)
