"""Operational logging must never copy speech content or credentials."""
from __future__ import annotations

import io
import json
import logging
import os
import stat

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from tts_api import operations, server
from managed_service import ManagedTTS


def fake_wav():
    out = io.BytesIO()
    sf.write(out, np.zeros(2400, dtype=np.float32), 48000, format="WAV")
    return out.getvalue()


@pytest.fixture
def activity_log(tmp_path, monkeypatch):
    # Restore process-wide log destination at the end of each test.
    monkeypatch.setattr(operations, "_FILE", None)
    operations.configure(tmp_path)
    return tmp_path / "operations.log"


def test_log_allowlist_discards_text_and_arbitrary_metadata(activity_log):
    private = "PERSONAL-SPOKEN-WORDS-ARE-NOT-FOR-LOGS"
    operation = operations.emit(
        "generation.started",
        model="voxcpm2", backend="voxcpm2",
        characters=27, has_reference=True,
        text=private, transcript=private, instructions=private,
        voice_description=private, error=private, path=private,
        error_type="RuntimeError",
    )
    data = activity_log.read_text(encoding="utf-8")
    assert private not in data
    assert "text" not in operation
    assert "transcript" not in operation
    assert "instructions" not in operation
    assert "voice_description" not in operation
    assert "error" not in operation
    assert operation["characters"] == 27
    assert operation["has_reference"] is True
    assert operation["model"] == "voxcpm2"
    assert operation["error_type"] == "RuntimeError"
    if os.name == "posix":
        assert stat.S_IMODE(activity_log.stat().st_mode) == 0o600


def test_unknown_event_and_model_identifiers_cannot_log_untrusted_text(activity_log):
    secret = "HIDDEN VOICE INSTRUCTION"
    with pytest.raises(ValueError, match="Unknown TTS operational event"):
        operations.emit(secret, text=secret)
    event = operations.emit("request.started", model=secret,
                            backend=secret, error_type=secret, status=secret)
    assert "model" not in event
    assert "backend" not in event
    assert "error_type" not in event
    assert "status" not in event
    assert secret not in activity_log.read_text()


def test_log_history_persists_and_truncates_to_requested_limit(activity_log):
    for i in range(4):
        operations.emit("model.load.started", model="voxcpm2", characters=i)
    assert [i["characters"] for i in operations.recent(2)] == [2, 3]
    assert len(operations.recent(200)) == 4


def test_request_events_redact_text_and_preserve_request_correlation(activity_log, monkeypatch, tmp_path):
    private = "SPEAK-THIS-WORD-FOR-WORD-NEVER-LOG-IT"
    monkeypatch.setattr(server, "STORE", server.VoiceStore(tmp_path / "voices"))
    monkeypatch.setattr(server, "_render", lambda request: fake_wav())
    result = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "voxcpm2", "input": private, "voice": "default",
        "response_format": "wav",
        "instructions": "DELICATE-DELIVERY-CONTENT-DO-NOT-LOG",
    })
    assert result.status_code == 200
    # Events are internal and go to the ComfyUI console, not the sidebar.
    assert TestClient(server.app).get("/api/operations?limit=100").status_code == 404
    history = operations.recent(100)
    assert any(x["event"] == "request.started" for x in history)
    assert any(x["event"] == "request.completed" for x in history)
    starts = [x for x in history if x["event"] == "request.started"]
    completions = [x for x in history if x["event"] == "request.completed"]
    assert starts[-1]["request_id"] == completions[-1]["request_id"]
    assert private not in activity_log.read_text()
    assert "DELICATE-DELIVERY-CONTENT" not in activity_log.read_text()


def test_failed_request_has_safe_status_not_backend_error_text(activity_log, monkeypatch):
    private = "THE-PRIVATE-INPUT-WAS-IN-EXCEPTION"
    def fail(_request):
        raise RuntimeError(private)
    monkeypatch.setattr(server, "_render", fail)
    response = TestClient(server.app).post("/v1/audio/speech", json={
        "model": "voxcpm2", "input": private, "response_format": "wav",
    })
    assert response.status_code == 503
    data = activity_log.read_text()
    assert private not in data
    events = operations.recent(100)
    assert any(entry["event"] == "request.failed" for entry in events)


def test_log_file_rotation_is_bounded(activity_log, monkeypatch):
    monkeypatch.setattr(operations, "MAX_BYTES", 300)
    for _ in range(40):
        operations.emit("model.load.started", model="voxcpm2")
    assert activity_log.stat().st_size <= 300
    assert (activity_log.parent / "operations.log.1").exists()
    assert len(list(activity_log.parent.glob("operations.log*"))) <= 3


def test_console_relay_shows_events_but_not_speech_text(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(operations, "_FILE", None)
    operations.configure(tmp_path / "data")
    secret = "PRIVATE SPOKEN WORDS NEVER TO APPEAR"
    operations.emit(
        "generation.started", backend="voxcpm2", model="voxcpm2",
        characters=len(secret), input=secret, instructions=secret,
        exception=secret, error_type="RuntimeError",
    )
    manager = ManagedTTS(root=tmp_path)
    with caplog.at_level(logging.INFO, logger="comfyui-tts-api"):
        manager._relay_events()
        manager._relay_events()  # Each line is printed once.
    lines = [record.getMessage() for record in caplog.records
             if record.name == "comfyui-tts-api"]
    assert len(lines) == 1
    assert "[TTS API] generation.started" in lines[0]
    assert "model=voxcpm2" in lines[0]
    assert "characters=" in lines[0]
    assert secret not in lines[0]


def test_console_relay_revalidates_tampered_log_entries(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(operations, "_FILE", None)
    operations.configure(tmp_path / "data")
    path = tmp_path / "data" / "operations.log"
    secret = "PRIVATE TTS INPUT"
    with path.open("a") as handle:
        handle.write(json.dumps({
            "event": "request.started", "model": secret,
            "error_type": secret, "characters": 33, "text": secret,
            "request_id": "not-a-valid-id",
        }) + "\n")
    manager = ManagedTTS(root=tmp_path)
    with caplog.at_level(logging.INFO, logger="comfyui-tts-api"):
        manager._relay_events()
    messages = [x.getMessage() for x in caplog.records
                if x.name == "comfyui-tts-api"]
    assert len(messages) == 1
    assert "request.started" in messages[0]
    assert "characters=33" in messages[0]
    assert secret not in messages[0]
    assert "not-a-valid-id" not in messages[0]


def test_cpu_int8_conversion_progress_relayed_without_speech_text(
        tmp_path, monkeypatch, caplog):
    """CPU/INT8 evidence reaches the ComfyUI console, not the private input."""
    monkeypatch.setattr(operations, "_FILE", None)
    operations.configure(tmp_path / "data")
    secret = "PRIVATE INPUT NEVER PRINTED"
    operations.emit("model.runtime.selected", model="voxcpm2",
                    backend="voxcpm2", device="cpu",
                    precision="int8", text=secret)
    operations.emit("model.quantization.started", backend="voxcpm2",
                    device="cpu", precision="int8",
                    quantized_layers=37, text=secret)
    operations.emit("model.quantization.completed", backend="voxcpm2",
                    device="cpu", precision="int8",
                    quantized_layers=37, text=secret)
    manager = ManagedTTS(root=tmp_path)
    with caplog.at_level(logging.INFO, logger="comfyui-tts-api"):
        manager._relay_events()
    lines = [item.getMessage() for item in caplog.records
             if item.name == "comfyui-tts-api"]
    assert any("model.runtime.selected" in item and "device=cpu" in item
               for item in lines)
    assert any("model.quantization.started" in item and
               "quantized_layers=37" in item for item in lines)
    assert any("model.quantization.completed" in item and
               "precision=int8" in item for item in lines)
    assert secret not in (tmp_path / "data" / "operations.log").read_text()
    assert not any(secret in item for item in lines)
