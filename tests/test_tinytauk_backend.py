from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tinytauk")

from tinytalk.backends.tinytauk import TinyTAuKEngine
from tinytalk.config import Settings


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
            audio=__import__("torch").zeros(int(24_000 * gen_seconds)),
            sample_rate=24_000,
        )


def _engine(**overrides):
    settings = Settings(
        backend="tinytauk",
        max_chars_per_chunk=28,
        inter_chunk_silence_ms=0,
        tinytauk_chars_per_second=14.0,
        **overrides,
    )
    engine = TinyTAuKEngine(settings)
    engine.tts = FakeTinyTAuK()
    engine.sample_rate = 24_000
    engine.loaded = True
    return engine


def test_instructions_are_composed_for_auk():
    engine = _engine()
    engine.synthesize("hello world", instructions="Speak calmly")

    call = engine.tts.calls[0]
    assert "Speak calmly" in call["instruction"]
    assert "hello world" in call["instruction"]


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
            import torch

            return SimpleNamespace(
                audio=torch.full((24_000,), 0.02, dtype=torch.float32),
                sample_rate=24_000,
            )

    engine.tts = QuietTinyTAuK()
    result = engine.synthesize("quiet speech")

    assert isinstance(result.audio, np.ndarray)
    assert float(np.max(result.audio)) <= 0.021
