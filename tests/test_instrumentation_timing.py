import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tinytalk.config import Settings
from tinytalk.engine import TinyTalkEngine


def _engine(*, max_retries: int = 0) -> TinyTalkEngine:
    settings = Settings(
        max_retries=max_retries,
        wer_endpoint="",
        wer_threshold=0.25,
        ref_codes=Path(__file__).parent / "voices" / "jo.pt",
        ref_text=Path(__file__).parent / "voices" / "jo.txt",
    )
    engine = TinyTalkEngine(settings)
    engine.tts = MagicMock()
    engine.ref_codes = [1, 2, 3]
    engine.ref_text = "test reference"
    engine.sample_rate = 24_000
    return engine


def _audio() -> np.ndarray:
    samples = 24_000
    return (0.08 * np.sin(np.linspace(0, 2 * np.pi * 440, samples))).astype(
        np.float32
    )


def test_failed_infer_attempt_keeps_elapsed_inference_time():
    engine = _engine(max_retries=1)
    calls = 0

    def infer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(0.01)
            raise ValueError("no speech tokens")
        return _audio()

    engine.tts.infer = infer

    with (
        patch.object(engine, "_apply_generation_settings"),
        patch.object(
            engine,
            "_chunk_wer",
            return_value=(0.1, "confidence", False),
        ),
    ):
        result = engine.synthesize("hello world")

    assert result.timing is not None
    attempts = result.timing.chunks[0]["attempts_detail"]
    assert len(attempts) == 2
    assert attempts[0]["wer"] is None
    assert attempts[0]["t_infer"] >= 0.008


def test_f0_timing_includes_pitch_analysis():
    engine = _engine()
    engine.tts.infer = MagicMock(return_value=_audio())

    def slow_f0(*args, **kwargs):
        time.sleep(0.01)
        return 440.0

    with (
        patch.object(engine, "_apply_generation_settings"),
        patch.object(
            engine,
            "_chunk_wer",
            return_value=(0.1, "confidence", False),
        ),
        patch("tinytalk.engine.compute_f0_mean", side_effect=slow_f0),
    ):
        result = engine.synthesize("hello world")

    assert result.timing is not None
    assert result.timing.chunks[0]["t_f0"] >= 0.008


def test_generation_settings_restore_even_when_request_fails():
    engine = _engine(max_retries=1)
    engine.tts.infer = MagicMock(side_effect=ValueError("no speech tokens"))

    with patch.object(engine, "_apply_generation_settings") as apply_settings:
        with pytest.raises(RuntimeError, match="produced no speech tokens"):
            engine.synthesize("hello world")

    assert len(apply_settings.call_args_list) == 3
    assert apply_settings.call_args_list[-1].args == (engine.tts,)
    assert apply_settings.call_args_list[-1].kwargs == {}
