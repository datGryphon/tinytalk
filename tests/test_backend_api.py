import builtins

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.config import Settings
from tinytalk.engine import SpeechEngine, SynthesisResult, TinyTalkEngine, create_engine


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


@pytest.mark.parametrize("backend", ["tinytauk", "omnivoice"])
def test_controlled_backend_receives_instructions_and_speed(monkeypatch, backend):
    engine = FakeControlledEngine()
    monkeypatch.setattr(server, "settings", Settings(backend=backend))
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
    assert response.headers["X-TinyTalk-Backend"] == backend
    assert response.headers["X-TinyTalk-Model"] == "fake/controlled"


@pytest.mark.parametrize("backend", ["neutts", "tinytauk", "omnivoice", "silencio2", "chatterbox_nano"])
def test_missing_backend_extra_has_actionable_error(monkeypatch, backend):
    original_import = builtins.__import__

    def missing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == f"backends.{backend}" and level == 1:
            raise ModuleNotFoundError(f"No module named {backend!r}", name=backend)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_import)

    expected = rf"uv sync --frozen --extra {backend}"
    with pytest.raises(RuntimeError, match=expected):
        create_engine(Settings(backend=backend))


def test_legacy_tinytalk_engine_respects_backend_selection():
    from tinytalk.backends.silencio2 import Silencio2Backend

    settings = Settings(backend="silencio2")

    engine = TinyTalkEngine(settings)

    assert type(engine) is SpeechEngine
    assert isinstance(engine.backend, Silencio2Backend)
    assert engine.settings is settings
