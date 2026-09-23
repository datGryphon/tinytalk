from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("tinytauk")

from tinytalk.backends import tinytauk as backend_module
from tinytalk.backends.tinytauk import TinyTAuKEngine
from tinytalk.config import Settings
from tinytalk.quality import QualityResult


class FakeTinyTAuK:
    def __init__(self):
        self.config = SimpleNamespace(runtime=SimpleNamespace(seed=100))
        self.calls = []

    def generate(self, instruction, *, gen_seconds, seed=None):
        self.calls.append(
            {
                "instruction": instruction,
                "gen_seconds": gen_seconds,
                "seed": seed,
            }
        )
        return SimpleNamespace(
            audio=torch.zeros(int(24_000 * gen_seconds)),
            sample_rate=24_000,
        )


def _engine(**overrides):
    values = {
        "backend": "tinytauk",
        "max_chars_per_chunk": 28,
        "inter_chunk_silence_ms": 0,
        "tinytauk_chars_per_second": 14.0,
        "max_retries": 0,
    }
    values.update(overrides)
    engine = TinyTAuKEngine(Settings(**values))
    engine.tts = FakeTinyTAuK()
    engine.sample_rate = 24_000
    engine.loaded = True
    return engine


def _quality(
    *,
    wer: float,
    cer: float | None = None,
    insertions: int = 0,
    deletions: int = 0,
    substitutions: int = 0,
    prompt_leak: bool = False,
    source: str = "whisper",
    fallback: bool = False,
) -> QualityResult:
    return QualityResult(
        wer=wer,
        cer=cer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        prompt_leak=prompt_leak,
        source=source,
        fallback=fallback,
    )


def test_load_uses_models_and_warms_full_generation(monkeypatch):
    fake = FakeTinyTAuK()
    factory_args = {}

    class FakeFactory:
        @staticmethod
        def from_pretrained(**kwargs):
            factory_args.update(kwargs)
            return fake

    monkeypatch.setattr(backend_module, "TinyTAuK", FakeFactory)
    settings = Settings(
        backend="tinytauk",
        tinytauk_model="example/AuK",
        tinytauk_qwen_model="example/Qwen",
    )
    engine = TinyTAuKEngine(settings)

    engine.load()

    assert factory_args == {
        "model_id": "example/AuK",
        "qwen_model_id": "example/Qwen",
    }
    assert fake.calls[0]["gen_seconds"] == pytest.approx(9.0)
    assert engine.loaded is True
    assert engine.sample_rate == 24_000


def test_load_uses_tinytauk_profile(monkeypatch, tmp_path):
    profile = tmp_path / "runtime.toml"
    profile.write_text("[model]\nmodel_id = 'example/AuK'\n")
    fake = FakeTinyTAuK()
    fake.config.model = SimpleNamespace(model_id="example/AuK")
    received = []

    class FakeFactory:
        @staticmethod
        def from_config(path):
            received.append(path)
            return fake

        @staticmethod
        def from_pretrained(**_kwargs):
            raise AssertionError("profile must override from_pretrained")

    monkeypatch.setattr(backend_module, "TinyTAuK", FakeFactory)
    engine = TinyTAuKEngine(
        Settings(
            backend="tinytauk",
            tinytauk_profile=profile,
            tinytauk_model="unused/default",
        )
    )

    engine.load()

    assert received == [profile]
    assert engine.model_name == "example/AuK"
    assert engine.loaded
    assert fake.calls[0]["gen_seconds"] == pytest.approx(9.0)


def test_instructions_use_canonical_auk_serialization():
    engine = _engine()
    engine.synthesize("hello world", instructions="Speak calmly")

    assert engine.tts.calls[0]["instruction"] == (
        'Based on the following description: "Speak calmly", '
        'generate speech content "hello world".'
    )


def test_default_description_uses_canonical_auk_serialization():
    engine = _engine()
    engine.synthesize("hello world")

    assert engine.tts.calls[0]["instruction"] == (
        'Based on the following description: "A clear, natural speaking voice", '
        'generate speech content "hello world".'
    )


def test_speed_scales_target_duration():
    engine = _engine()
    text = "x" * 28

    engine.synthesize(text, speed=2.0)

    assert engine.tts.calls[0]["gen_seconds"] == pytest.approx(1.0)


def test_chunk_seeds_are_deterministic_and_distinct():
    engine = _engine()
    engine.synthesize("one two three four five six seven eight nine ten")

    assert len(engine.tts.calls) >= 2
    assert [call["seed"] for call in engine.tts.calls[:2]] == [100, 101]


def test_backend_does_not_loudness_normalize():
    engine = _engine()

    class QuietTinyTAuK(FakeTinyTAuK):
        def generate(self, instruction, *, gen_seconds, seed=None):
            self.calls.append(
                {"instruction": instruction, "gen_seconds": gen_seconds, "seed": seed}
            )
            return SimpleNamespace(
                audio=torch.full((24_000,), 0.02, dtype=torch.float32),
                sample_rate=24_000,
            )

    engine.tts = QuietTinyTAuK()
    result = engine.synthesize("quiet speech")

    assert isinstance(result.audio, np.ndarray)
    assert float(np.max(result.audio)) <= 0.021


def test_prompt_leak_rerolls_with_new_seed_and_same_horizon(monkeypatch):
    engine = _engine(max_retries=2, wer_endpoint="http://asr.local")
    scores = iter(
        [
            _quality(
                wer=0.7,
                cer=0.5,
                insertions=4,
                substitutions=3,
                prompt_leak=True,
            ),
            _quality(wer=0.0, cer=0.0),
        ]
    )
    monkeypatch.setattr(backend_module, "evaluate_audio", lambda *args, **kwargs: next(scores))

    result = engine.synthesize("x" * 28, instructions="Speak calmly and deliberately")

    calls = engine.tts.calls
    assert [call["seed"] for call in calls] == [100, 101]
    assert calls[0]["gen_seconds"] == pytest.approx(2.0)
    assert calls[1]["gen_seconds"] == pytest.approx(2.0)

    chunk = result.timing.chunks[0]
    assert chunk["attempts"] == 2
    assert chunk["attempts_detail"][0]["prompt_leak"] is True
    assert chunk["attempts_detail"][0]["status"] is None
    assert chunk["attempts_detail"][1]["status"] == "accepted"
    assert chunk["status"] == "accepted"
    assert chunk["seed"] == 101


def test_non_leak_insertions_shrink_generation_horizon(monkeypatch):
    engine = _engine(max_retries=1, wer_endpoint="http://asr.local")
    scores = iter(
        [
            _quality(wer=0.5, cer=0.3, insertions=3),
            _quality(wer=0.0, cer=0.0),
        ]
    )
    monkeypatch.setattr(backend_module, "evaluate_audio", lambda *args, **kwargs: next(scores))

    engine.synthesize("x" * 28)

    assert engine.tts.calls[0]["gen_seconds"] == pytest.approx(2.0)
    assert engine.tts.calls[1]["gen_seconds"] == pytest.approx(1.84)


def test_deletions_grow_generation_horizon(monkeypatch):
    engine = _engine(max_retries=1, wer_endpoint="http://asr.local")
    scores = iter(
        [
            _quality(wer=0.5, cer=0.3, deletions=3),
            _quality(wer=0.0, cer=0.0),
        ]
    )
    monkeypatch.setattr(backend_module, "evaluate_audio", lambda *args, **kwargs: next(scores))

    engine.synthesize("x" * 28)

    assert engine.tts.calls[0]["gen_seconds"] == pytest.approx(2.0)
    assert engine.tts.calls[1]["gen_seconds"] == pytest.approx(2.2)


def test_retry_seed_ranges_do_not_collide_between_chunks(monkeypatch):
    engine = _engine(max_retries=2, wer_endpoint="http://asr.local")
    monkeypatch.setattr(
        backend_module,
        "evaluate_audio",
        lambda *args, **kwargs: _quality(wer=1.0, cer=1.0, substitutions=1),
    )

    engine.synthesize("one two three four five six seven eight nine ten")

    seeds = [call["seed"] for call in engine.tts.calls]
    assert seeds[:6] == [100, 101, 102, 103, 104, 105]


def test_accepted_retry_wins_over_lower_ranked_failed_candidate(monkeypatch):
    engine = _engine(max_retries=1, wer_endpoint="http://asr.local")
    scores = iter(
        [
            _quality(wer=0.3, cer=0.2, substitutions=1),
            _quality(wer=0.1, cer=None, source="confidence", fallback=True),
        ]
    )
    monkeypatch.setattr(backend_module, "evaluate_audio", lambda *args, **kwargs: next(scores))

    result = engine.synthesize("x" * 28)

    chunk = result.timing.chunks[0]
    assert chunk["status"] == "accepted"
    assert chunk["seed"] == 101
    assert chunk["wer"] == pytest.approx(0.1)
    assert chunk["attempts_detail"][1]["status"] == "accepted"


def test_exhausted_retries_keep_best_non_leaking_candidate(monkeypatch):
    engine = _engine(max_retries=2, wer_endpoint="http://asr.local")
    scores = iter(
        [
            _quality(wer=0.05, cer=0.02, insertions=3, prompt_leak=True),
            _quality(wer=0.6, cer=0.4, substitutions=2),
            _quality(wer=0.3, cer=0.2, substitutions=1),
        ]
    )
    monkeypatch.setattr(backend_module, "evaluate_audio", lambda *args, **kwargs: next(scores))

    result = engine.synthesize("x" * 28)

    chunk = result.timing.chunks[0]
    selected = [detail for detail in chunk["attempts_detail"] if detail["status"] is not None]
    assert len(selected) == 1
    assert selected[0]["attempt"] == 2
    assert selected[0]["status"] == "fallback"
    assert chunk["status"] == "fallback"
    assert chunk["prompt_leak"] is False
    assert chunk["wer"] == pytest.approx(0.3)
