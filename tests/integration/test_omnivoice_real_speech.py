import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("omnivoice")

from tinytalk import server
from tinytalk.backends.omnivoice import OmniVoiceEngine
from tinytalk.config import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("TINYTALK_RUN_OMNIVOICE_INTEGRATION") != "1",
    reason="set TINYTALK_RUN_OMNIVOICE_INTEGRATION=1 to run real OmniVoice integration tests",
)


def test_omnivoice_voice_design_smoke(monkeypatch):
    settings = Settings(
        backend="omnivoice",
        omnivoice_model=os.getenv("TINYTALK_OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
        omnivoice_device=os.getenv("TINYTALK_OMNIVOICE_DEVICE", "cpu"),
        omnivoice_language=os.getenv("TINYTALK_OMNIVOICE_LANGUAGE", "en"),
        max_chars_per_chunk=180,
        max_retries=0,
    )
    monkeypatch.setattr(server, "settings", settings)
    monkeypatch.setattr(server, "engine", OmniVoiceEngine(settings))

    artifact_dir = Path("test_artifacts")
    artifact_dir.mkdir(exist_ok=True)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "TinyTalk is now speaking through OmniVoice.",
                "instructions": "A calm, natural technical narrator",
                "speed": 1.0,
                "response_format": "wav",
            },
            timeout=1800,
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["X-TinyTalk-Backend"] == "omnivoice"
    assert int(response.headers["X-TinyTalk-Chunks"]) == 1

    out = artifact_dir / "omnivoice-smoke.wav"
    out.write_bytes(response.content)
    assert out.stat().st_size > 44
