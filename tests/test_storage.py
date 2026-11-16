from tts_api.storage import VoiceStore
import pytest


def test_voice_roundtrip(tmp_path):
    store = VoiceStore(tmp_path)
    record = store.create(name="Alice", description="Soft voice")
    assert store.get(record["id"]) == record
    assert len(store.list()) == 1
    assert store.delete(record["id"])
    assert store.get(record["id"]) is None


def test_reject_traversal(tmp_path):
    store = VoiceStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../../secrets")
