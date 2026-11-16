import numpy as np
import pytest

import torch

from tinytalk.backends import chatterbox_nano as backend
from tinytalk.engine import SpeechEngine
from tinytalk.config import Settings


class Model:
    """Fake upstream model: 24 kHz, native float tensor output, recorded calls."""

    def __init__(self):
        self.sr = 24000
        self.generate_calls = []
        self.prepare_calls = []

    def generate(self, text, **kwargs):
        self.generate_calls.append((text, kwargs))
        return torch.full((240,), 0.1)

    def prepare_conditionals(self, path, **kwargs):
        self.prepare_calls.append((path, kwargs))


def _engine(monkeypatch, model, **overrides):
    values = {
        "backend": "chatterbox_nano",
        "max_chars_per_chunk": 30,
        "inter_chunk_silence_ms": 0,
    }
    values.update(overrides)
    loads = []

    def from_pretrained(**kwargs):
        loads.append(kwargs)
        return model

    monkeypatch.setattr(
        backend.ChatterboxTurboTTS, "from_pretrained", staticmethod(from_pretrained)
    )
    return SpeechEngine(backend.ChatterboxNanoBackend(Settings(**values))), model, loads


@pytest.fixture
def engine(monkeypatch):
    model = Model()
    instance, _, loads = _engine(monkeypatch, model)
    instance.load()
    instance.load()
    assert len(loads) == 1
    assert loads[0] == {"device": "cpu", "nano": True}
    return instance


def test_native_output_and_timing(engine):
    result = engine.synthesize("Hello.")
    np.testing.assert_allclose(result.audio, 0.1)
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert result.timing.chunks[0]["attempts"] == 1
    assert "text" not in result.timing.chunks[0]
    assert result.timing.chunks[0]["status"] == "accepted"
    assert "t_infer" in result.timing.chunks[0]
    assert engine.backend.tts.generate_calls[0][0] == result.chunks[0]
    assert engine.backend.tts.generate_calls[0][1] == {}


def test_multi_chunk_timing_and_assembly(engine):
    result = engine.synthesize("One two three. Four five six. Seven eight nine.")
    assert len(result.chunks) >= 2
    assert len(result.timing.chunks) == len(result.chunks)
    assert result.audio.dtype == np.float32
    assert len(result.audio) == 240 * len(result.chunks)


def test_load_caches_same_model(monkeypatch):
    model = Model()
    instance, _, loads = _engine(monkeypatch, model)
    instance.load()
    assert instance.loaded is True
    first = instance.backend.tts
    assert len(loads) == 1
    instance.load()
    assert instance.backend.tts is first
    assert instance.sample_rate == 24000
    assert len(loads) == 1


def test_reference_prepared_once_with_native_levels(monkeypatch, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"")
    model = Model()
    instance, _, loads = _engine(
        monkeypatch, model, chatterbox_nano_ref_audio=ref
    )
    instance.load()
    instance.load()
    assert len(loads) == 1
    assert len(model.prepare_calls) == 1
    path, kwargs = model.prepare_calls[0]
    assert path == str(ref)
    assert kwargs == {"exaggeration": 0.0, "norm_loudness": False}


def test_missing_reference_rejected(monkeypatch, tmp_path):
    model = Model()
    instance, _, loads = _engine(
        monkeypatch, model, chatterbox_nano_ref_audio=tmp_path / "missing.wav"
    )
    with pytest.raises(ValueError):
        instance.load()
    assert len(loads) == 0


def test_restores_different_tags_in_order(engine):
    result = engine.synthesize(
        "First sentence. [laugh] Second sentence. [sigh] Last sentence."
    )
    assert len(result.chunks) >= 2
    joined = " ".join(result.chunks)
    assert joined.count("[laugh]") == 1
    assert joined.count("[sigh]") == 1
    assert joined.index("[laugh]") < joined.index("[sigh]")
    assert "\uE000" not in joined


def test_reserved_marker_rejected(engine):
    with pytest.raises(ValueError):
        engine.synthesize("\uE000 hello")


def test_oversized_tag_rejected(engine):
    big = "[" + "x" * 31 + "]"
    with pytest.raises(ValueError):
        engine.synthesize("hello " + big)


@pytest.mark.parametrize("kwargs", [{"instructions": "happy"}, {"speed": 1.5}])
def test_unsupported_controls(engine, kwargs):
    with pytest.raises(ValueError):
        engine.synthesize("Hello", **kwargs)


def test_empty_text_rejected(engine):
    with pytest.raises(ValueError):
        engine.synthesize("   ")


def test_unloaded_rejected():
    engine = SpeechEngine(backend.ChatterboxNanoBackend(Settings(backend="chatterbox_nano")))
    with pytest.raises(RuntimeError):
        engine.synthesize("Hello")
