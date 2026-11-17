"""Opt-in HTTP qualification using external models and reference audio."""

import io
import os
from pathlib import Path
import wave

import pytest
from fastapi.testclient import TestClient

from tinytalk.config import Settings
from tinytalk.engine import create_engine


@pytest.mark.parametrize("backend", ["silencio2", "chatterbox_nano"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_resident_expressive_http(backend, conditioned, monkeypatch):
    if os.getenv("TINYTALK_RUN_EXPRESSIVE_INTEGRATION") != backend:
        pytest.skip("select one expressive backend explicitly")
    reference = os.getenv("TINYTALK_TEST_REF_AUDIO")
    if conditioned and not reference:
        pytest.skip("external reference audio required")
    from tinytalk import server

    options = {"backend": backend, "max_chars_per_chunk": 80}
    if conditioned:
        if backend == "silencio2":
            options["silencio2_voice"] = reference
        else:
            options["chatterbox_nano_ref_audio"] = Path(reference)
    settings = Settings(**options)
    engine = create_engine(settings)
    monkeypatch.setattr(server, "settings", settings)
    monkeypatch.setattr(server, "engine", engine)
    output = Path(os.getenv("TINYTALK_TEST_ARTIFACTS", "test_artifacts")) / backend
    output.mkdir(parents=True, exist_ok=True)
    mode = "reference" if conditioned else "bundled"
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        model = engine.backend.tts
        cached = dict(model._voices) if backend == "silencio2" else model.conds
        for name, text in [
            ("ordinary", "The system is operating normally."),
            ("events", "That was surprising. [laugh] Let us continue."),
            ("long", ("<whisper> " if backend == "silencio2" else "") + "That was surprising. [laugh] Let us continue. " * 4),
        ]:
            response = client.post("/v1/audio/speech", json={"input": text, "voice": "ignored"})
            assert response.status_code == 200
            with wave.open(io.BytesIO(response.content)) as audio:
                assert audio.getframerate() == 24000
                assert audio.getnchannels() == 1
                assert audio.getnframes() > 0
            if name == "long":
                assert int(response.headers["X-TinyTalk-Chunks"]) > 1
            (output / f"{mode}-{name}.wav").write_bytes(response.content)
            assert engine.backend.tts is model
            if backend == "silencio2":
                assert model._voices.keys() == cached.keys()
                assert all(model._voices[key] is value for key, value in cached.items())
            else:
                assert model.conds is cached
