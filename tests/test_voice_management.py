"""Saved voice inspection and editing contracts; no model or GPU dependencies."""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tts_api import server
from tts_api.storage import VoiceStore


def recording(freq=220, sample_rate=24000):
    t = np.arange(9600) / sample_rate
    sample = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wav = io.BytesIO()
    sf.write(wav, sample, sample_rate, format="WAV", subtype="PCM_16")
    return wav.getvalue()


def test_edit_voice_preserves_id_audio_and_unrecognized_metadata(tmp_path):
    store = VoiceStore(tmp_path)
    created = store.create(
        name="Character A", description="Original description",
        reference=recording(), reference_text="Hello",
    )
    old_audio = store.audio_path(created).read_bytes()
    updated = store.update(
        created["id"], name="Character B", description="Soft and quiet",
        reference_text="Corrected text",
    )
    assert updated["id"] == created["id"]
    assert updated["reference_file"] == created["reference_file"]
    assert updated["created_at"] == created["created_at"]
    assert updated["updated_at"] != ""
    assert store.get(created["id"]) == updated
    assert store.audio_path(updated).read_bytes() == old_audio


def test_edit_reference_replaces_wav_in_place_not_voice_id(tmp_path):
    store = VoiceStore(tmp_path)
    original = store.create(name="Original", reference=recording(freq=210))
    old_data = store.audio_path(original).read_bytes()
    replacement = recording(freq=440)
    edited = store.update(
        original["id"], name="New name", description="",
        reference_text="", reference=replacement,
    )
    assert edited["id"] == original["id"]
    assert edited["reference_file"] == original["reference_file"]
    assert store.audio_path(edited).read_bytes() == replacement
    assert store.audio_path(edited).read_bytes() != old_data
    if os.name == "posix":
        assert store.audio_path(edited).stat().st_mode & 0o777 == 0o600
        assert store._path(edited["id"]).stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.tmp"))
    assert store.delete(original["id"])
    assert not list(tmp_path.glob("*.wav"))


def test_voice_api_read_edit_and_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", VoiceStore(tmp_path))
    monkeypatch.setattr(server, "KEY", "")
    client = TestClient(server.app)
    audio = recording()
    created = client.post(
        "/api/voices/clone",
        data={"name": "First", "description": "", "reference_text": "First words",
              "consent": "true"},
        files={"audio": ("sample.wav", audio, "audio/wav")},
    )
    assert created.status_code == 200, created.text
    voice_id = created.json()["id"]
    details = client.get(f"/api/voices/{voice_id}")
    assert details.status_code == 200
    assert details.json()["name"] == "First"
    sample = client.get(f"/api/voices/{voice_id}/audio")
    assert sample.status_code == 200
    assert sample.headers["content-type"].startswith("audio/wav")
    assert sample.headers["cache-control"] == "no-store"
    assert sample.content.startswith(b"RIFF")
    updated = client.patch(f"/api/voices/{voice_id}", data={
        "name": "Renamed", "description": "Expressive character",
        "reference_text": "Updated words",
    })
    assert updated.status_code == 200, updated.text
    assert updated.json()["id"] == voice_id
    assert updated.json()["name"] == "Renamed"
    assert updated.json()["description"] == "Expressive character"
    assert updated.json()["reference_text"] == "Updated words"
    assert client.get(f"/api/voices/{voice_id}/audio").content == sample.content
    assert any(item["id"] == voice_id and item["name"] == "Renamed"
               for item in client.get("/api/voices").json())


def test_replace_recording_requires_permission_and_valid_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", VoiceStore(tmp_path))
    monkeypatch.setattr(server, "KEY", "")
    client = TestClient(server.app)
    created = server.STORE.create(name="First", reference=recording(freq=220),
                                  reference_text="Original")
    voice_id = created["id"]
    endpoint = f"/api/voices/{voice_id}"
    original_audio = client.get(endpoint + "/audio").content
    form = {"name": "Updated", "description": "Modified",
            "reference_text": "Updated transcript"}
    no_permission = client.patch(endpoint, data=form,
        files={"audio": ("replacement.wav", recording(freq=440), "audio/wav")})
    assert no_permission.status_code == 422
    assert server.STORE.get(voice_id)["name"] == "First"
    bad = client.patch(endpoint, data={**form, "consent": "true"},
        files={"audio": ("invalid.wav", b"not audio", "audio/wav")})
    assert bad.status_code == 422
    assert server.STORE.get(voice_id)["name"] == "First"
    assert client.get(endpoint + "/audio").content == original_audio
    success = client.patch(endpoint, data={**form, "consent": "true"},
        files={"audio": ("replacement.wav", recording(freq=440), "audio/wav")})
    assert success.status_code == 200, success.text
    assert success.json()["id"] == voice_id
    assert success.json()["name"] == "Updated"
    assert success.json()["reference_text"] == "Updated transcript"
    assert client.get(endpoint + "/audio").content != original_audio


@pytest.mark.parametrize("name,description,reference_text", [
    ("", "", ""),
    (" " * 5, "", ""),
    ("x" * 121, "", ""),
    ("Valid", "d" * 1001, ""),
    ("Valid", "", "r" * 4001),
])
def test_voice_edits_reject_bad_metadata_without_changes(
        tmp_path, monkeypatch, name, description, reference_text):
    monkeypatch.setattr(server, "STORE", VoiceStore(tmp_path))
    client = TestClient(server.app)
    saved = server.STORE.create(name="Original")
    result = client.patch(f"/api/voices/{saved['id']}", data={
        "name": name,
        "description": description,
        "reference_text": reference_text,
    })
    assert result.status_code == 422
    assert server.STORE.get(saved["id"])["name"] == "Original"


def test_missing_voice_or_audio_and_unsafe_id(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", VoiceStore(tmp_path))
    client = TestClient(server.app)
    assert client.get("/api/voices/voice_missing").status_code == 404
    assert client.get("/api/voices/voice_missing/audio").status_code == 404
    assert client.patch("/api/voices/voice_missing", data={"name": "Nope"}).status_code == 404
    without_audio = server.STORE.create(name="No reference")
    assert client.get(f"/api/voices/{without_audio['id']}/audio").status_code == 404
    assert client.get("/api/voices/%2E%2E%2Fsecret").status_code in (404, 422)


def test_voice_management_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", VoiceStore(tmp_path))
    monkeypatch.setattr(server, "KEY", "test-api-key")
    saved = server.STORE.create(name="Private", reference=recording())
    client = TestClient(server.app)
    path = f"/api/voices/{saved['id']}"
    for method, endpoint, data in [
        ("get", path, None),
        ("get", path + "/audio", None),
        ("patch", path, {"name": "Updated"}),
    ]:
        reply = getattr(client, method)(endpoint, **({"data": data} if data else {}))
        assert reply.status_code == 401
    authorized = client.get(path + "/audio",
                            headers={"Authorization": "Bearer test-api-key"})
    assert authorized.status_code == 200


def test_legacy_voice_records_remain_editable(tmp_path):
    voice_id = "voice_1234567890abcdef"
    (tmp_path / (voice_id + ".json")).write_text(
        '{"id": "' + voice_id + '", "name": "Legacy", "description": "", '
        '"reference_text": "", "reference_file": null}',
        encoding="utf-8",
    )
    store = VoiceStore(tmp_path)
    saved = store.update(
        voice_id, name="Updated legacy", description="Old file still works",
        reference_text="",
    )
    assert saved["id"] == voice_id
    assert saved["name"] == "Updated legacy"
    assert "created_at" not in saved
    assert "updated_at" in saved
