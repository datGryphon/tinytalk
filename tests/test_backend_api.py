import numpy as np
import pytest
from fastapi.testclient import TestClient

from tinytalk import engine as engine_module
from tinytalk import server
from tinytalk.config import Settings
from tinytalk.engine import SynthesisResult, TinyTalkEngine, create_engine


class FakeControlledEngine:
    loaded = True
    model_name = "fake/controlled"
    last_instructions = None
    last_speed = None

    def __init__(self, settings=None):
        self.settings = settings

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


def test_missing_backend_extra_has_actionable_error(monkeypatch):
    def missing_import(_module_name):
        raise ModuleNotFoundError("No module named 'tinytauk'", name="tinytauk")

    monkeypatch.setattr(engine_module, "import_module", missing_import)

    with pytest.raises(RuntimeError, match=r"pip install 'tinytalk\[tinytauk\]'"):
        create_engine(Settings(backend="tinytauk"))


def test_legacy_tinytalk_engine_respects_backend_selection(monkeypatch):
    seen = []

    def fake_backend_class(backend):
        seen.append(backend)
        return FakeControlledEngine

    monkeypatch.setattr(engine_module, "_backend_class", fake_backend_class)
    settings = Settings(backend="tinytauk")

    engine = TinyTalkEngine(settings)

    assert isinstance(engine, FakeControlledEngine)
    assert engine.settings is settings
    assert seen == ["tinytauk"]
