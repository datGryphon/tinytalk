from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omnivoice")

from tinytalk.backends import omnivoice as omnivoice_backend
from tinytalk.config import Settings
from tinytalk.quality import QualityResult


class FakeOmniVoiceModel:
    sampling_rate = 24_000

    def __init__(self):
        self.generate_calls = []
        self.clone_calls = []

    def create_voice_clone_prompt(self, *, ref_audio, ref_text):
        self.clone_calls.append({"ref_audio": ref_audio, "ref_text": ref_text})
        return {"cached": True}

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [np.full(2400, 0.1, dtype=np.float32)]


class FakeOmniVoice:
    model = FakeOmniVoiceModel()
    load_kwargs = None

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.model = FakeOmniVoiceModel()
        cls.load_kwargs = {"model_name": model_name, **kwargs}
        return cls.model


def good_quality() -> QualityResult:
    return QualityResult(
        wer=0.0,
        cer=0.0,
        substitutions=0,
        deletions=0,
        insertions=0,
        prompt_leak=False,
        source="whisper",
        fallback=False,
    )


def bad_quality() -> QualityResult:
    return QualityResult(
        wer=0.5,
        cer=0.5,
        substitutions=1,
        deletions=0,
        insertions=0,
        prompt_leak=False,
        source="whisper",
        fallback=False,
    )


def make_engine(monkeypatch, settings: Settings) -> omnivoice_backend.OmniVoiceEngine:
    monkeypatch.setattr(omnivoice_backend, "OmniVoice", FakeOmniVoice)
    monkeypatch.setattr(
        omnivoice_backend,
        "evaluate_audio",
        lambda *args, **kwargs: good_quality(),
    )
    engine = omnivoice_backend.OmniVoiceEngine(settings)
    engine.load()
    return engine


def test_loads_cpu_model_in_float32(monkeypatch):
    engine = make_engine(monkeypatch, Settings(backend="omnivoice"))

    assert engine.loaded
    assert FakeOmniVoice.load_kwargs["model_name"] == "k2-fsa/OmniVoice"
    assert FakeOmniVoice.load_kwargs["device_map"] == "cpu"
    assert FakeOmniVoice.load_kwargs["dtype"] is omnivoice_backend.torch.float32


def test_auto_voice_passes_speed_and_language(monkeypatch):
    settings = Settings(backend="omnivoice", omnivoice_language="en")
    engine = make_engine(monkeypatch, settings)

    result = engine.synthesize("hello world", speed=1.4)

    call = FakeOmniVoice.model.generate_calls[-1]
    assert call["language"] == "en"
    assert call["speed"] == 1.4
    assert call["instruct"] is None
    assert call["voice_clone_prompt"] is None
    assert result.timing.chunks[0]["mode"] == "auto"
    assert result.timing.chunks[0]["status"] == "accepted"


def test_instructions_select_voice_design(monkeypatch):
    engine = make_engine(monkeypatch, Settings(backend="omnivoice"))

    result = engine.synthesize(
        "hello world",
        instructions="female, young adult, british accent",
    )

    call = FakeOmniVoice.model.generate_calls[-1]
    assert call["instruct"] == "female, young adult, british accent"
    assert call["voice_clone_prompt"] is None
    assert result.timing.chunks[0]["mode"] == "design"


def test_clone_prompt_is_cached_and_used(monkeypatch, tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_text = tmp_path / "ref.txt"
    ref_audio.write_bytes(b"RIFF")
    ref_text.write_text("Reference transcript.", encoding="utf-8")
    settings = Settings(
        backend="omnivoice",
        omnivoice_ref_audio=ref_audio,
        omnivoice_ref_text=ref_text,
    )
    engine = make_engine(monkeypatch, settings)

    result = engine.synthesize("hello world")

    assert FakeOmniVoice.model.clone_calls == [
        {"ref_audio": str(ref_audio), "ref_text": "Reference transcript."}
    ]
    call = FakeOmniVoice.model.generate_calls[-1]
    assert call["voice_clone_prompt"] == {"cached": True}
    assert call["instruct"] is None
    assert result.timing.chunks[0]["mode"] == "clone"


def test_design_instruction_overrides_configured_clone(monkeypatch, tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_text = tmp_path / "ref.txt"
    ref_audio.write_bytes(b"RIFF")
    ref_text.write_text("Reference transcript.", encoding="utf-8")
    settings = Settings(
        backend="omnivoice",
        omnivoice_ref_audio=ref_audio,
        omnivoice_ref_text=ref_text,
    )
    engine = make_engine(monkeypatch, settings)

    engine.synthesize("hello world", instructions="female, elderly, low pitch")

    call = FakeOmniVoice.model.generate_calls[-1]
    assert call["instruct"] == "female, elderly, low pitch"
    assert call["voice_clone_prompt"] is None


def test_upstream_instruction_validation_errors_are_not_swallowed(monkeypatch):
    engine = make_engine(monkeypatch, Settings(backend="omnivoice"))

    def invalid_generate(**_kwargs):
        raise ValueError("Unsupported instruct items")

    FakeOmniVoice.model.generate = invalid_generate

    with pytest.raises(ValueError, match="Unsupported instruct items"):
        engine.synthesize("hello world", instructions="calm technical narrator")


def test_reference_audio_and_text_must_be_paired(monkeypatch, tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(omnivoice_backend, "OmniVoice", FakeOmniVoice)
    engine = omnivoice_backend.OmniVoiceEngine(
        Settings(backend="omnivoice", omnivoice_ref_audio=ref_audio)
    )

    with pytest.raises(ValueError, match="must be configured together"):
        engine.load()


def test_quality_retry_adds_sampling_temperature(monkeypatch):
    monkeypatch.setattr(omnivoice_backend, "OmniVoice", FakeOmniVoice)
    qualities = iter([bad_quality(), good_quality()])
    monkeypatch.setattr(
        omnivoice_backend,
        "evaluate_audio",
        lambda *args, **kwargs: next(qualities),
    )
    engine = omnivoice_backend.OmniVoiceEngine(
        Settings(backend="omnivoice", max_retries=1)
    )
    engine.load()

    result = engine.synthesize("hello world")

    calls = FakeOmniVoice.model.generate_calls
    assert [call["class_temperature"] for call in calls] == [0.0, 0.2]
    timing = result.timing.chunks[0]
    assert timing["attempts"] == 2
    assert timing["status"] == "accepted"
    assert timing["attempts_detail"][0]["status"] is None
    assert timing["attempts_detail"][1]["status"] == "accepted"


def test_quality_exhaustion_returns_best_candidate_as_fallback(monkeypatch):
    monkeypatch.setattr(omnivoice_backend, "OmniVoice", FakeOmniVoice)
    monkeypatch.setattr(
        omnivoice_backend,
        "evaluate_audio",
        lambda *args, **kwargs: bad_quality(),
    )
    engine = omnivoice_backend.OmniVoiceEngine(
        Settings(backend="omnivoice", max_retries=1)
    )
    engine.load()

    result = engine.synthesize("hello world")

    timing = result.timing.chunks[0]
    assert timing["attempts"] == 2
    assert timing["status"] == "fallback"
    assert sum(
        detail["status"] == "fallback" for detail in timing["attempts_detail"]
    ) == 1
