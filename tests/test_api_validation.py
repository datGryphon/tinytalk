import numpy as np
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.engine import SynthesisResult


class FakeEngine:
    loaded = True
    last_text = None

    def synthesize(self, text):
        self.last_text = text
        return SynthesisResult(
            audio=np.zeros(2400, dtype=np.float32),
            sample_rate=24_000,
            chunks=[text],
        )

    def load(self):
        self.loaded = True


def test_rejects_unknown_response_format(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "response_format": "flac"},
        )
    assert res.status_code == 400


def test_rejects_stream(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "hello", "stream": True})
    assert res.status_code == 400


def test_rejects_blank_input(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "   "})
    assert res.status_code == 400


def test_strips_input_before_synthesis(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "  hello  "})
    assert res.status_code == 200
    assert engine.last_text == "hello"


def test_accepts_ignored_openai_fields_and_returns_wav(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={
                "input": "hello",
                "model": "ignored",
                "voice": "ignored",
                "speed": 1.5,
                "unknown": "ignored",
            },
        )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.headers["X-TinyTalk-Chunks"] == "1"
