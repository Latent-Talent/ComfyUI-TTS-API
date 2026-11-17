"""ComfyUI queue tickets for the separate managed TTS worker.

Internal node is the single real ComfyUI queue item for each speech request.
No spoken text, voice description or API credentials enter the prompt queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import logging
import re
import threading
import time
import uuid

LOG = logging.getLogger("comfyui-tts-api")
TICKET_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SLOT_MAX_SECONDS = 1800


@dataclass
class Slot:
    ticket: str
    prompt_id: str
    created: float = field(default_factory=time.monotonic)
    state: str = "pending"
    done: threading.Event = field(default_factory=threading.Event)


class QueueBroker:
    def __init__(self):
        self._lock = threading.RLock()
        self._slots: dict[str, Slot] = {}
        self._manager = None
        self._queue = None

    def configure(self, manager):
        self._manager = manager

    def active(self) -> bool:
        with self._lock:
            return any(slot.state in ("activating", "running", "releasing")
                       for slot in self._slots.values())

    def outstanding(self) -> bool:
        with self._lock:
            return any(slot.state in ("pending", "activating", "running") for slot in self._slots.values())

    def create(self, ticket: str, server, kind: str = "speech") -> str:
        if not isinstance(ticket, str) or not TICKET_PATTERN.fullmatch(ticket):
            raise ValueError("Invalid TTS queue ticket")
        if kind not in ("speech", "voice_design"):
            raise ValueError("Unsupported TTS queue operation")
        queue = getattr(server, "prompt_queue", None)
        if queue is None:
            raise RuntimeError("ComfyUI prompt queue is unavailable")
        with self._lock:
            self._queue = queue
            if ticket in self._slots:
                raise ValueError("TTS queue ticket already exists")
            prompt_id = str(uuid.uuid4())
            self._slots[ticket] = Slot(ticket=ticket, prompt_id=prompt_id)
            # Private node is not part of the user's graph; it reserves one
            # actual queued execution, with no audio or transcript in inputs.
            prompt = {
                "1": {
                    "class_type": "LatentTalentTTSQueueSlot",
                    "inputs": {"ticket": ticket},
                    "_meta": {"title": "TTS · " + ("Speech" if kind == "speech" else "Voice design")},
                }
            }
            try:
                number = server.number
                server.number += 1
                queue.put((
                    number, prompt_id, prompt,
                    {"comfy_usage_source": "comfyui-tts-api",
                     "create_time": int(time.time() * 1000)},
                    ["1"], {},
                ))
            except Exception:
                self._slots.pop(ticket, None)
                raise
            LOG.info("[TTS API] Queued %s reservation %s", kind, prompt_id)
            return prompt_id

    def state(self, ticket: str, queue=None) -> str:
        with self._lock:
            slot = self._slots.get(ticket)
            if slot is None:
                return "cancelled"
            if slot.done.is_set():
                return "completed"
            if slot.state == "pending" and queue is not None:
                running, pending = queue.get_current_queue_volatile()
                identifiers = {item[1] for item in running + pending}
                if slot.prompt_id not in identifiers:
                    slot.state = "cancelled"
                    slot.done.set()
            return slot.state

    def release(self, ticket: str) -> bool:
        with self._lock:
            slot = self._slots.get(ticket)
            if slot is None:
                return False
            was_pending = slot.state in ("pending", "cancelled")
            slot.done.set()
            if slot.state != "cancelled":
                slot.state = "completed" if was_pending else "releasing"
            if was_pending:
                # Never leave an abandoned waiting job in ComfyUI's queue.
                if self._queue is not None:
                    self._queue.delete_queue_item(lambda item: item[1] == slot.prompt_id)
                self._slots.pop(ticket, None)
            return True

    def execute(self, ticket: str) -> tuple:
        with self._lock:
            slot = self._slots.get(ticket)
            if slot is None or slot.done.is_set():
                return ()
            slot.state = "activating"
        manager = self._manager
        # The worker might be paused from an earlier ComfyUI workflow.
        # Do not admit speech until it has resumed and this queue item runs.
        if manager is not None and manager._running():
            if not manager._request("/api/internal/resume", timeout=4):
                LOG.warning("[TTS API] Could not resume TTS for queue reservation")
                self.release(ticket)
                with self._lock:
                    self._slots.pop(ticket, None)
                return ()
        with self._lock:
            if slot.done.is_set():
                self._slots.pop(ticket, None)
                return ()
            slot.state = "running"
        started = time.monotonic()
        LOG.info("[TTS API] TTS reservation running %s", slot.prompt_id)
        try:
            while not slot.done.wait(0.2):
                if time.monotonic() - started > SLOT_MAX_SECONDS:
                    LOG.warning("[TTS API] Queue reservation timed out")
                    if manager is not None:
                        manager._terminate()
                    break
                if manager is not None and not manager._running():
                    LOG.warning("[TTS API] TTS worker stopped during queue reservation")
                    break
                try:
                    import nodes
                    nodes.before_node_execution()
                except ImportError:
                    pass
                except Exception:
                    # ComfyUI's interrupt flag is global. Terminate our owned
                    # worker before freeing this queue slot, to avoid two GPU
                    # jobs running at the same time.
                    if manager is not None:
                        manager._terminate()
                    raise
        finally:
            with self._lock:
                slot.done.set()
                slot.state = "completed"
                self._slots.pop(ticket, None)
            LOG.info("[TTS API] TTS queue slot released %s", slot.prompt_id)
        return ()


BROKER = QueueBroker()


class TTSQueueSlot:
    """Internal queue reservation; never carries speech content."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ticket": ("STRING", {"default": ""})}}

    RETURN_TYPES = ()
    FUNCTION = "reserve"
    OUTPUT_NODE = True
    CATEGORY = "TTS API/internal"
    DESCRIPTION = "Internal TTS queue reservation. Not intended for manual use."

    @classmethod
    def IS_CHANGED(cls, ticket):
        return float("nan")

    def reserve(self, ticket):
        return BROKER.execute(ticket)
