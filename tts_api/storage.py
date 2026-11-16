"""Persistent local voice metadata and immutable reference audio."""
from __future__ import annotations
import json
import os
import re
import uuid
from pathlib import Path

VALID_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


class VoiceStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, voice_id: str) -> Path:
        if not VALID_ID.fullmatch(voice_id):
            raise ValueError("Invalid voice id.")
        return self.root / f"{voice_id}.json"

    def list(self) -> list[dict]:
        result = []
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
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def create(self, name: str, description: str = "", reference: bytes | None = None,
               reference_text: str = "") -> dict:
        voice_id = "voice_" + uuid.uuid4().hex[:16]
        record = {"id": voice_id, "name": name, "description": description,
                  "reference_text": reference_text, "reference_file": None}
        if reference is not None:
            target = self.root / f"{voice_id}.wav"
            with target.open("xb") as fh:
                fh.write(reference)
            record["reference_file"] = target.name
        target = self._path(voice_id)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temp, target)
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
        record = self.get(voice_id)
        if record is None:
            return False
        path = self.audio_path(record)
        self._path(voice_id).unlink()
        if path:
            path.unlink(missing_ok=True)
        return True
