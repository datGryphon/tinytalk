import numpy as np
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.config import Settings
from tinytalk.engine import SynthesisResult


class FakeControlledEngine:
    loaded = True
    model_name = "fake/controlled"
    last_instructions = None
    last_speed = None

    def load(self):
        self.loaded = True

    def synthesize(self, text, *, instructions=None, speed=None):
        self.last_instructions = instructions
        self.last_speed = speed
        return SynthesisResult(
            audio=np.zeros(2400, dtype=np.float32),
            sample_rate=24_000,
            chunks=[text],
        )


def test_rejects_speed_outside_supported_range(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeControlledEngine())
    with TestClient(server.app) as client:
        too_slow = client.post("/v1/audio/speech", json={"input": "hello", "speed": 0.1})
        too_fast = client.post("/v1/audio/speech", json={"input": "hello", "speed": 5.0})
    assert too_slow.status_code == 400
    assert too_fast.status_code == 400


def test_controlled_backend_receives_instructions_and_speed(monkeypatch):
    engine = FakeControlledEngine()
    monkeypatch.setattr(server, "settings", Settings(backend="tinytauk"))
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "hello",
                "instructions": "  Speak calmly.  ",
                "speed": 1.5,
            },
        )
    assert response.status_code == 200
    assert engine.last_instructions == "Speak calmly."
    assert engine.last_speed == 1.5
    assert response.headers["X-TinyTalk-Backend"] == "tinytauk"
    assert response.headers["X-TinyTalk-Model"] == "fake/controlled"
