import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("tinytauk")

from tinytalk import server
from tinytalk.backends.tinytauk import TinyTAuKEngine
from tinytalk.config import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("TINYTALK_RUN_TINYTAUK_INTEGRATION") != "1",
    reason="set TINYTALK_RUN_TINYTAUK_INTEGRATION=1 to run real TinyTAuK integration tests",
)


def test_tinytauk_speech_smoke(monkeypatch):
    settings = Settings(backend="tinytauk", max_chars_per_chunk=180)
    monkeypatch.setattr(server, "settings", settings)
    monkeypatch.setattr(server, "engine", TinyTAuKEngine(settings))

    artifact_dir = Path("test_artifacts")
    artifact_dir.mkdir(exist_ok=True)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "TinyTalk is now speaking through TinyTAuK.",
                "instructions": "A calm, natural technical narrator",
                "response_format": "wav",
            },
            timeout=1800,
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["X-TinyTalk-Backend"] == "tinytauk"
    assert int(response.headers["X-TinyTalk-Chunks"]) == 1

    out = artifact_dir / "tinytauk-smoke.wav"
    out.write_bytes(response.content)
    assert out.stat().st_size > 44
