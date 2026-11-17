"""Persistent local voice metadata and reference recordings.

A voice ID remains stable when its display fields or audio are updated,
preserving existing SillyTavern assignments and model-default references.
"""
from __future__ import annotations
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

VALID_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VoiceStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, voice_id: str) -> Path:
        if not VALID_ID.fullmatch(voice_id):
            raise ValueError("Invalid voice id.")
        return self.root / f"{voice_id}.json"

    def _save_record(self, record: dict) -> None:
        """Atomic metadata replacement, with no names or transcripts in logs."""
        target = self._path(record["id"])
        tmp = self.root / f".{record['id']}.{uuid.uuid4().hex}.tmp"
        try:
            with tmp.open("x", encoding="utf-8") as fh:
                os.chmod(tmp, 0o600)
                json.dump(record, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def list(self) -> list[dict]:
        result = []
        with self._lock:
            for item in sorted(self.root.glob("*.json")):
                try:
                    value = json.loads(item.read_text(encoding="utf-8"))
                    if value.get("id") == item.stem:
                        result.append(value)
                except (OSError, ValueError):
                    continue
        return result

    def get(self, voice_id: str) -> dict | None:
        path = self._path(voice_id)
        with self._lock:
            if not path.exists():
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("id") != voice_id:
                raise ValueError("Voice record ID mismatch")
            return value

    def create(self, name: str, description: str = "", reference: bytes | None = None,
               reference_text: str = "") -> dict:
        voice_id = "voice_" + uuid.uuid4().hex[:16]
        record = {"id": voice_id, "name": name, "description": description,
                  "reference_text": reference_text, "reference_file": None,
                  "created_at": _now()}
        with self._lock:
            if reference is not None:
                target = self.root / f"{voice_id}.wav"
                with target.open("xb") as fh:
                    fh.write(reference)
                record["reference_file"] = target.name
            try:
                self._save_record(record)
            except BaseException:
                if reference is not None:
                    (self.root / f"{voice_id}.wav").unlink(missing_ok=True)
                raise
        return record

    def update(self, voice_id: str, *, name: str, description: str,
               reference_text: str, reference: bytes | None = None) -> dict | None:
        """Update a saved voice in place; optional audio is replaced atomically."""
        with self._lock:
            record = self.get(voice_id)
            if record is None:
                return None
            record = dict(record)
            record.update(name=name, description=description,
                          reference_text=reference_text, updated_at=_now())
            if reference is not None:
                target = self.root / f"{voice_id}.wav"
                tmp = self.root / f".{voice_id}.{uuid.uuid4().hex}.wav.tmp"
                try:
                    with tmp.open("xb") as fh:
                        fh.write(reference)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
                record["reference_file"] = target.name
            self._save_record(record)
            return record

    def audio_path(self, record: dict) -> Path | None:
        filename = record.get("reference_file")
        if not filename:
            return None
        expected = record["id"] + ".wav"
        if filename != expected:
            raise ValueError("Invalid voice reference.")
        path = self.root / expected
        if not path.is_file():
            raise FileNotFoundError(expected)
        return path

    def delete(self, voice_id: str) -> bool:
        with self._lock:
            record = self.get(voice_id)
            if record is None:
                return False
            try:
                path = self.audio_path(record)
            except FileNotFoundError:
                path = None
            self._path(voice_id).unlink()
            if path:
                path.unlink(missing_ok=True)
            return True
