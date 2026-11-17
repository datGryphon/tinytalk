import json

import numpy as np
import pytest
from pydantic import ValidationError

from tinytalk.config import Settings, VoiceConfig
from tinytalk.engine import SpeechBackend, SpeechEngine, SynthesisRequest
from tinytalk.timing import ChunkTiming


class FakeBackend(SpeechBackend[SynthesisRequest]):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "fake")
        self.loaded = True
        self.seen: list[str | None] = []

    def load(self) -> None:
        self.loaded = True

    def prepare_request(
        self, *, instructions: str | None, speed: float | None
    ) -> SynthesisRequest:
        return SynthesisRequest(instructions=instructions, speed=speed)

    def synthesize_chunk(
        self, request: SynthesisRequest, chunk_text: str, index: int, num_chunks: int
    ):
        self.seen.append(request.voice.description if request.voice else None)
        audio = np.ones(240, dtype=np.float32) * 0.1
        return audio, ChunkTiming(index=index, attempts=1, duration=0.01, status="accepted"), 0


def voice_settings(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({
        "backend": "tinytauk",
        "default": "narrator",
        "voices": {
            "narrator": {"description": "clear", "max_chars_per_chunk": 12},
            "dramatic": {"description": "dramatic", "inter_chunk_silence_ms": 0},
        },
    }))
    return Settings(backend="tinytauk", voices_file=path)


def test_default_and_explicit_voice(tmp_path):
    backend = FakeBackend(voice_settings(tmp_path))
    engine = SpeechEngine(backend)
    engine.synthesize("Hello.", voice=None)
    engine.synthesize("Hello.", voice="dramatic")
    assert backend.seen == ["clear", "dramatic"]
    with pytest.raises(ValueError, match="unknown voice"):
        engine.synthesize("Hello.", voice="missing")


def test_legacy_ignores_voice():
    backend = FakeBackend(Settings(backend="tinytauk"))
    engine = SpeechEngine(backend)
    engine.synthesize("Hello.", voice="unconfigured")
    assert backend.seen == [None]


def test_voice_config_rejects_wrong_fields_and_missing_default():
    with pytest.raises(ValidationError, match="default voice"):
        VoiceConfig.model_validate({
            "backend": "tinytauk", "default": "absent",
            "voices": {"speaker": {"description": "clear"}},
        })
    with pytest.raises(ValidationError, match="unsupported fields"):
        VoiceConfig.model_validate({
            "backend": "silencio2", "default": "speaker",
            "voices": {"speaker": {"voice_id": "260", "ref_audio": "/tmp/x.wav"}},
        })
    with pytest.raises(ValidationError, match="requires ref_codes and ref_text"):
        VoiceConfig.model_validate({
            "backend": "neutts", "default": "speaker",
            "voices": {"speaker": {"ref_codes": "/tmp/ref.pt"}},
        })


def test_per_voice_silence_does_not_leak(tmp_path):
    backend = FakeBackend(voice_settings(tmp_path))
    engine = SpeechEngine(backend)
    text = "Hello. Another sentence."
    silent = engine.synthesize(text, voice="dramatic")
    normal = engine.synthesize(text, voice="narrator")
    assert len(normal.audio) >= len(silent.audio)
    assert engine.settings.inter_chunk_silence_ms == 60


def test_api_selects_default_and_rejects_unknown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tinytalk import server

    backend = FakeBackend(voice_settings(tmp_path))
    monkeypatch.setattr(server, "engine", SpeechEngine(backend))
    with TestClient(server.app) as client:
        response = client.post("/v1/audio/speech", json={"input": "Hello."})
        assert response.status_code == 200
        assert backend.seen[-1] == "clear"

        response = client.post(
            "/v1/audio/speech", json={"input": "Hello.", "voice": "dramatic"}
        )
        assert response.status_code == 200
        assert backend.seen[-1] == "dramatic"

        response = client.post(
            "/v1/audio/speech", json={"input": "Hello.", "voice": "missing"}
        )
        assert response.status_code == 400
