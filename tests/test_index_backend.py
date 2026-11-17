"""IndexTTS integration and manual, non-automatic model migration tests."""
from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tts_api import server
from tts_api.models import INDEX_ID, INDEX_EMOTIONS, canonical_model, model_catalog
from tts_api.model_storage import model_root


def silent_wav():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(4800, dtype=np.float32), 24000, format="WAV")
    return buffer.getvalue()


class FakeIndexEngine:
    def __init__(self):
        self.loaded = False
        self.paused = False
        self.last_error = None
        self.last_stats = {"seconds": 0.03, "audio_seconds": 0.2}
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return silent_wav()

    def unload(self):
        self.loaded = False

    def try_pause(self):
        self.paused = True
        return True

    def resume(self):
        self.paused = False

    def unload_if_idle(self, seconds):
        return False


def test_index_model_metadata_and_canonical_aliases():
    assert len(INDEX_EMOTIONS) == 8
    assert INDEX_EMOTIONS == (
        "happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm",
    )
    assert canonical_model("indextts") == INDEX_ID
    assert canonical_model(INDEX_ID) == INDEX_ID
    assert any(m["id"] == INDEX_ID for m in model_catalog())
    assert canonical_model("tts-1", INDEX_ID) == INDEX_ID


def test_saved_voice_routes_to_index_and_manual_emotion_vector(tmp_path, monkeypatch):
    store = server.VoiceStore(tmp_path / "voices")
    record = store.create(name="Actor", reference=silent_wav())
    fake = FakeIndexEngine()
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "INDEX_ENGINE", fake)
    result = TestClient(server.app).post("/v1/audio/speech", json={
        "model": INDEX_ID, "voice": record["id"], "input": "Speak this line",
        "response_format": "wav", "language": "English",
        "emotion_mode": "vector", "emotion_vector": [0.8, 0, 0, 0, 0, 0, 0, 0.2],
        "emotion_strength": 0.7, "duration_factor": 1.2,
    })
    assert result.status_code == 200, result.text
    assert result.content.startswith(b"RIFF")
    assert len(fake.calls) == 1
    _, kwargs = fake.calls[0]
    assert kwargs["reference_path"] == str(store.audio_path(record))
    assert kwargs["emotion_mode"] == "vector"
    assert kwargs["emotion_vector"] == [0.8, 0, 0, 0, 0, 0, 0, 0.2]
    assert kwargs["duration_factor"] == 1.2


def test_index_instructions_use_text_emotion_and_second_reference(tmp_path, monkeypatch):
    store = server.VoiceStore(tmp_path / "voices")
    speaker = store.create(name="Actor", reference=silent_wav())
    emotion = store.create(name="Emotional delivery", reference=silent_wav())
    fake = FakeIndexEngine()
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "INDEX_ENGINE", fake)
    client = TestClient(server.app)
    response = client.post("/v1/audio/speech", json={
        "model": INDEX_ID, "voice": speaker["id"], "input": "Hello",
        "response_format": "wav", "instructions": "Say it angrily",
    })
    assert response.status_code == 200, response.text
    assert fake.calls[-1][1]["emotion_mode"] == "text"

    response = client.post("/v1/audio/speech", json={
        "model": INDEX_ID, "voice": speaker["id"], "input": "Hello",
        "response_format": "wav", "emotion_mode": "reference",
        "emotion_reference_voice": emotion["id"],
    })
    assert response.status_code == 200, response.text
    assert fake.calls[-1][1]["emotion_reference"] == str(store.audio_path(emotion))


def test_index_without_saved_voice_fails_before_model_load(tmp_path, monkeypatch):
    fake = FakeIndexEngine()
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path / "voices"))
    monkeypatch.setattr(server, "INDEX_ENGINE", fake)
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": INDEX_ID, "voice": "default", "input": "Hello",
        "response_format": "wav",
    })
    assert response.status_code == 422
    assert fake.calls == []


def test_model_directory_is_fixed_path_with_no_legacy_migration(tmp_path, monkeypatch):
    path = tmp_path / "ComfyUI" / "models" / "TTS"
    monkeypatch.setenv("TTS_MODELS_DIR", str(path))
    legacy = tmp_path / "old-cache" / "models--openbmb--VoxCPM2"
    legacy.mkdir(parents=True)
    (legacy / "keep.bin").write_bytes(b"keep-me")
    assert model_root() == path
    assert (legacy / "keep.bin").read_bytes() == b"keep-me"


def test_move_script_is_explicit_and_preserves_hf_snapshot_links(tmp_path):
    script = Path(__file__).resolve().parents[1] / "move-tts-models.sh"
    home = tmp_path / "home"
    src = home / ".cache" / "huggingface" / "hub"
    repo_name = "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    model = src / repo_name
    (model / "blobs").mkdir(parents=True)
    (model / "snapshots" / "hash").mkdir(parents=True)
    (model / "blobs" / "1").write_bytes(b"weights")
    (model / "snapshots" / "hash" / "config.json").symlink_to("../../blobs/1")
    unrelated = src / "models--other--unrelated"
    unrelated.mkdir()
    (unrelated / "data").write_bytes(b"unchanged")
    dst = tmp_path / "ComfyUI" / "models" / "TTS"
    env = dict(os.environ)
    env.update({"HOME": str(home), "HF_HUB_CACHE": str(src), "TTS_MODELS_DIR": str(dst)})
    dry = subprocess.run(["bash", str(script), "--dry-run"], env=env,
                         capture_output=True, text=True, check=True)
    assert "DRY RUN" in dry.stdout
    assert model.is_dir()
    assert not (dst / "huggingface" / "hub" / repo_name).exists()
    subprocess.run(["bash", str(script)], env=env,
                   capture_output=True, text=True, check=True)
    target = dst / "huggingface" / "hub" / repo_name
    assert not model.exists()
    assert (target / "snapshots" / "hash" / "config.json").read_bytes() == b"weights"
    assert (unrelated / "data").read_bytes() == b"unchanged"
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert target.is_dir()
