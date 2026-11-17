"""Privacy-preserving TTS operational events.

This is intentionally NOT a generic logging helper: only a fixed allowlist of
event names and scalar metadata can be persisted or returned to browsers.
Never pass transcript, input, instructions, descriptions, audio paths, API keys,
exception messages, or traceback strings to the event log.
"""
from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading

LOGGER = logging.getLogger("uvicorn.error")
REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tts_operation_id", default=None
)
_LOCK = threading.RLock()
_FILE: Path | None = None
MAX_BYTES = 2 * 1024 * 1024
BACKUPS = 2

EVENTS = frozenset({
    "service.started", "service.stopped",
    "request.started", "request.completed", "request.failed",
    "voice.design.started", "voice.design.completed", "voice.design.failed",
    "voice.clone.saved", "voice.deleted",
    "model.load.started", "model.load.completed", "model.load.failed",
    "model.unloaded", "model.switch", "model.idle_unloaded",
    "generation.started", "generation.completed", "generation.failed",
    "style.reference.started", "style.reference.cached",
    "style.reference.completed",
    "download.started", "download.completed", "download.failed",
    "worker.paused", "worker.resumed", "worker.pause_blocked",
})

# Only these keys may cross the log boundary. No arbitrary values or keys.
INT_FIELDS = {"characters", "audio_bytes", "sample_rate", "steps"}
FLOAT_FIELDS = {"elapsed_ms", "audio_seconds", "rtf"}
BOOL_FIELDS = {"cached", "has_reference", "has_instructions"}
STRING_FIELDS = {
    "model": r"^[a-zA-Z0-9_.-]{1,80}$",
    "backend": r"^(voxcpm2|qwen3-tts)$",
    "voice_type": r"^(default|preset|saved|openai_alias)$",
    "status": r"^(ok|error|paused|idle|starting|running)$",
    "error_type": r"^[a-zA-Z][a-zA-Z0-9_]{0,90}$",
    "format": r"^(wav|mp3|opus|aac|flac|pcm)$",
    "kind": r"^(speech|voice_design|voice_clone|download)$",
    "reason": r"^(api|idle|switch|workflow|shutdown|manual|fallback|restart)$",
}
ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")
VALID_STRINGS = {key: re.compile(regex) for key, regex in STRING_FIELDS.items()}


def configure(directory: Path) -> None:
    """Configure a file containing only sanitized events; never raw worker output."""
    global _FILE
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _LOCK:
        _FILE = directory / "operations.log"
        fd = os.open(_FILE, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
        if os.name == "posix":
            os.chmod(_FILE, 0o600)


def emit(event: str, **metadata) -> dict:
    if event not in EVENTS:
        raise ValueError("Unknown TTS operational event")
    item = {
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
    }
    request_id = REQUEST_ID.get()
    if isinstance(request_id, str) and ID_PATTERN.fullmatch(request_id):
        item["request_id"] = request_id
    for key, value in metadata.items():
        if key in INT_FIELDS and type(value) is int:
            item[key] = max(0, min(value, 10**12))
        elif key in FLOAT_FIELDS and type(value) in (int, float):
            if 0 <= value < 10**10:
                item[key] = round(float(value), 3)
        elif key in BOOL_FIELDS and type(value) is bool:
            item[key] = value
        elif (key in VALID_STRINGS and type(value) is str
              and VALID_STRINGS[key].fullmatch(value)):
            item[key] = value
        # Other keys and unsupported values are deliberately discarded.
    line = json.dumps(item, ensure_ascii=True, separators=(",", ":"))
    with _LOCK:
        if _FILE is not None:
            try:
                if _FILE.exists() and _FILE.stat().st_size + len(line) + 1 > MAX_BYTES:
                    for n in range(BACKUPS, 0, -1):
                        current = _FILE.with_name("operations.log." + str(n))
                        predecessor = (_FILE if n == 1
                                       else _FILE.with_name("operations.log." + str(n - 1)))
                        if predecessor.exists():
                            os.replace(predecessor, current)
                fd = os.open(_FILE, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                # Log storage failures must never interrupt an audio job.
                pass
    LOGGER.info("[TTS EVENT] %s", line)
    return item


def recent(limit: int = 100) -> list[dict]:
    """Bounded recent-event tail; never opens the raw worker log."""
    limit = max(1, min(int(limit), 200))
    with _LOCK:
        path = _FILE
        if path is None or not path.exists():
            return []
        # O(64 KiB) memory regardless of log file size.
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - 65536))
            raw = handle.read(65536)
        lines = raw.splitlines()
        if size > 65536:
            lines = lines[1:]
    records = []
    for raw_line in lines[-limit:]:
        try:
            event = json.loads(raw_line)
            if isinstance(event, dict) and event.get("event") in EVENTS:
                # Revalidate on read to protect against manually edited files.
                meta = {k: event[k] for k in INT_FIELDS | FLOAT_FIELDS |
                        BOOL_FIELDS | set(STRING_FIELDS) if k in event}
                filtered = {"time": event.get("time", ""), "event": event["event"]}
                if isinstance(event.get("request_id"), str) and ID_PATTERN.fullmatch(event["request_id"]):
                    filtered["request_id"] = event["request_id"]
                for key, value in meta.items():
                    if key in INT_FIELDS and type(value) is int:
                        filtered[key] = value
                    elif key in FLOAT_FIELDS and type(value) in (float, int):
                        filtered[key] = value
                    elif key in BOOL_FIELDS and type(value) is bool:
                        filtered[key] = value
                    elif key in VALID_STRINGS and type(value) is str and VALID_STRINGS[key].fullmatch(value):
                        filtered[key] = value
                records.append(filtered)
        except (ValueError, UnicodeDecodeError):
            continue
    return records
