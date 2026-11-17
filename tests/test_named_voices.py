import json
from typing import override

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tinytalk import server
from tinytalk.config import (
    NeuTTSVoice, Settings, SilencioVoice, TinyTAuKVoice, VoiceConfig, load_voice_config,
)
from tinytalk.engine import ChunkResult, SpeechBackend, SpeechEngine, SynthesisRequest
from tinytalk.timing import ChunkTiming


class FakeBackend(SpeechBackend[SynthesisRequest]):
    seen: list[str | None]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "fake", load_voice_config(settings, TinyTAuKVoice))
        self.loaded = True
        self.seen = []

    @override
    def load(self) -> None:
        self.loaded = True

    @override
    def prepare_request(
        self, *, instructions: str | None, speed: float | None, voice: str | None = None
    ) -> SynthesisRequest:
        return SynthesisRequest(instructions=instructions, speed=speed, voice=voice)

    @override
    def synthesize_chunk(
        self, request: SynthesisRequest, chunk_text: str, index: int, num_chunks: int
    ) -> ChunkResult:
        self.seen.append(request.voice)
        audio = np.ones(240, dtype=np.float32) * 0.1
        return audio, ChunkTiming(index=index, attempts=1, duration=0.01, status="accepted"), 0


def voice_settings(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({
        "backend": "tinytauk",
        "default": "narrator",
        "voices": {
            "narrator": {"description": "clear", "max_chars_per_chunk": 12},
            "dramatic": {"description": "dramatic", "max_chars_per_chunk": 12, "inter_chunk_silence_ms": 0},
            "long": {"description": "long"},
        },
    }))
    return Settings(backend="tinytauk", voices_file=path)


def test_default_and_explicit_voice(tmp_path):
    backend = FakeBackend(voice_settings(tmp_path))
    engine = SpeechEngine(backend)
    engine.synthesize("Hello.", voice=None)
    engine.synthesize("Hello.", voice="dramatic")
    assert backend.seen == ["narrator", "dramatic"]
    with pytest.raises(ValueError, match="unknown voice"):
        engine.synthesize("Hello.", voice="missing")


def test_legacy_ignores_voice():
    backend = FakeBackend(Settings(backend="tinytauk"))
    engine = SpeechEngine(backend)
    engine.synthesize("Hello.", voice="unconfigured")
    assert backend.seen == [None]


def test_voice_config_rejects_wrong_fields_and_missing_default():
    with pytest.raises(ValidationError, match="default voice"):
        VoiceConfig[TinyTAuKVoice].model_validate({
            "backend": "tinytauk", "default": "absent",
            "voices": {"speaker": {"description": "clear"}},
        })
    with pytest.raises(ValidationError, match="Extra inputs"):
        VoiceConfig[SilencioVoice].model_validate({
            "backend": "silencio2", "default": "speaker",
            "voices": {"speaker": {"voice_id": "260", "ref_audio": "/tmp/x.wav"}},
        })
    with pytest.raises(ValidationError, match="ref_text"):
        VoiceConfig[NeuTTSVoice].model_validate({
            "backend": "neutts", "default": "speaker",
            "voices": {"speaker": {"ref_codes": "/tmp/ref.pt"}},
        })


def test_rejects_backend_mismatch(tmp_path):
    settings = voice_settings(tmp_path).model_copy(update={"backend": "omnivoice"})
    with pytest.raises(ValueError, match="backend does not match"):
        load_voice_config(settings, TinyTAuKVoice)


def test_rejects_missing_named_reference(tmp_path):
    with pytest.raises(ValidationError, match="Path does not point to a file"):
        VoiceConfig[NeuTTSVoice].model_validate({
            "backend": "neutts", "default": "speaker",
            "voices": {"speaker": {
                "ref_codes": tmp_path / "missing.pt", "ref_text": tmp_path / "missing.txt",
            }},
        })


def test_per_voice_silence_does_not_leak(tmp_path):
    backend = FakeBackend(voice_settings(tmp_path))
    engine = SpeechEngine(backend)
    text = "Hello. Another sentence."
    silent = engine.synthesize(text, voice="dramatic")
    normal = engine.synthesize(text, voice="narrator")
    long = engine.synthesize(text, voice="long")
    assert normal.chunks == silent.chunks
    assert len(normal.chunks) > 1
    assert len(normal.audio) - len(silent.audio) == (len(normal.chunks) - 1) * 1440
    assert long.chunks == [text]
    assert engine.settings.inter_chunk_silence_ms == 60


def test_api_selects_default_and_rejects_unknown(tmp_path, monkeypatch):
    backend = FakeBackend(voice_settings(tmp_path))
    monkeypatch.setattr(server, "engine", SpeechEngine(backend))
    with TestClient(server.app) as client:
        response = client.post("/v1/audio/speech", json={"input": "Hello."})
        assert response.status_code == 200
        assert backend.seen[-1] == "narrator"

        response = client.post(
            "/v1/audio/speech", json={"input": "Hello.", "voice": "dramatic"}
        )
        assert response.status_code == 200
        assert backend.seen[-1] == "dramatic"

        response = client.post(
            "/v1/audio/speech", json={"input": "Hello.", "voice": "missing"}
        )
        assert response.status_code == 400
