import sys
from types import SimpleNamespace

import numpy as np
import pytest

import torch

from tinytalk.backends import silencio2 as backend
from tinytalk.engine import SpeechEngine
from tinytalk.config import Settings


class Model:
    sample_rate = 24000

    def __init__(self):
        self.calls = []
        self.voices = []

    def voice(self, voice):
        self.voices.append(voice)

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return torch.full((240,), 0.1)


@pytest.fixture
def engine(monkeypatch, tmp_path):
    model = Model()
    loads = []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return model

    module = SimpleNamespace(
        __file__=str(tmp_path / "silencio2/__init__.py"),
        EMOTIONS=("whisper", "happy", "sad"), SOUNDS=("laugh", "sigh", "clear_throat"),
        Silencio2TTS=SimpleNamespace(from_pretrained=load),
    )
    monkeypatch.setitem(sys.modules, "silencio2", module)
    monkeypatch.setattr(backend, "snapshot_download", lambda **kwargs: str(tmp_path))
    instance = SpeechEngine(backend.Silencio2Backend(Settings(backend="silencio2", max_chars_per_chunk=30)))
    instance.load()
    instance.load()
    assert len(loads) == 1
    assert loads[0][1] == {"device": "cpu"}
    assert model.voices == ["260"]
    return instance


def test_native_output_and_timing(engine):
    result = engine.synthesize("Hello.")
    np.testing.assert_allclose(result.audio, 0.1)
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert result.timing.chunks[0]["attempts"] == 1
    assert "text" not in result.timing.chunks[0]
    assert engine.backend.tts.calls[0][1] == {"voice": "260", "normalize": False}


def test_restores_different_tags_in_order(engine):
    result = engine.synthesize("First sentence. [laugh] Second sentence. [sigh] Last sentence.")
    joined = " ".join(result.chunks)
    assert joined.count("[laugh]") == joined.count("[sigh]") == 1
    assert joined.index("[laugh]") < joined.index("[sigh]")
    assert "\uE000" not in joined


def test_nonleading_emotion_persists(engine):
    chunks = engine.backend._carry_emotions(["Hello <happy> there <sad> now", "Next sentence."])
    assert chunks[1] == "<sad> Next sentence."


@pytest.mark.parametrize("text", ["[unknown]", "<unknown> Hello", "\uE000 hello", " "])
def test_invalid_input(engine, text):
    with pytest.raises(ValueError):
        engine.synthesize(text)


@pytest.mark.parametrize("kwargs", [{"instructions": "whisper"}, {"speed": 1.5}])
def test_unsupported_controls(engine, kwargs):
    with pytest.raises(ValueError):
        engine.synthesize("Hello", **kwargs)


def test_unloaded_rejected():
    with pytest.raises(RuntimeError):
        SpeechEngine(backend.Silencio2Backend(Settings())).synthesize("Hello")
