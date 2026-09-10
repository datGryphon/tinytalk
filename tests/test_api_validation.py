import numpy as np
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.engine import SynthesisResult


class FakeEngine:
    loaded = True
    model_name = "fake/model"
    last_text = None
    last_instructions = None
    last_speed = None

    def synthesize(self, text, *, instructions=None, speed=None):
        self.last_text = text
        self.last_instructions = instructions
        self.last_speed = speed
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


def test_rejects_speed_outside_supported_range(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        too_slow = client.post("/v1/audio/speech", json={"input": "hello", "speed": 0.1})
        too_fast = client.post("/v1/audio/speech", json={"input": "hello", "speed": 5.0})
    assert too_slow.status_code == 400
    assert too_fast.status_code == 400


def test_strips_input_before_synthesis(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "  hello  "})
    assert res.status_code == 200
    assert engine.last_text == "hello"


def test_forwards_instructions_and_speed(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={
                "input": "hello",
                "instructions": "  Speak calmly.  ",
                "speed": 1.5,
            },
        )
    assert res.status_code == 200
    assert engine.last_instructions == "Speak calmly."
    assert engine.last_speed == 1.5


def test_accepts_ignored_fields_and_returns_wav(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={
                "input": "hello",
                "model": "ignored",
                "voice": "ignored",
                "unknown": "ignored",
            },
        )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.headers["X-TinyTalk-Chunks"] == "1"
    assert res.headers["X-TinyTalk-Model"] == "fake/model"
