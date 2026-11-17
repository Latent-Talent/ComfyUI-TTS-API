"""Worker side of the real ComfyUI queue reservation protocol.

No speech text, voice descriptions, audio, or credentials are sent as queue
inputs. Only an opaque random ticket and operation kind cross the boundary.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
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


class QueueClient:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url if url is not None else os.getenv("TTS_COMFY_QUEUE_URL", "")).rstrip("/")
        self.token = token if token is not None else os.getenv("TTS_CONTROL_TOKEN", "")

    def _post(self, endpoint: str, ticket: str, kind: str = "speech") -> dict:
        if not self.url or not self.token:
            raise QueueUnavailable("Managed TTS queue bridge is unconfigured")
        data = json.dumps({"ticket": ticket, "kind": kind}).encode("ascii")
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

    def release(self, ticket: str | None):
        if ticket:
            self._post("release", ticket)


@asynccontextmanager
async def queued_tts(kind: str):
    client = QueueClient()
    ticket = await anyio.to_thread.run_sync(client.acquire, kind)
    try:
        yield
    finally:
        if ticket:
            try:
                # Release even if audio encoding or synthesis failed. Avoid
                # suppressing the original generation error if release fails.
                await anyio.to_thread.run_sync(client.release, ticket)
            except QueueUnavailable:
                pass
