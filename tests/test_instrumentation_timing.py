import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tinytalk.config import Settings
from tinytalk.backends.neutts import NeuTTSBackend, NeuTTSRequest
from tinytalk.engine import SpeechEngine
from tinytalk.quality import QualityResult

_ACCEPTED_QUALITY = QualityResult(0.1, None, None, None, None, False, "confidence", False)


def _engine(*, max_retries: int = 0) -> SpeechEngine[NeuTTSRequest]:
    settings = Settings(
        max_retries=max_retries,
        wer_endpoint="",
        wer_threshold=0.25,
        ref_codes=Path(__file__).parent / "voices" / "jo.pt",
        ref_text=Path(__file__).parent / "voices" / "jo.txt",
    )
    engine = SpeechEngine(NeuTTSBackend(settings))
    engine.backend.tts = MagicMock()
    engine.backend.ref_codes = [1, 2, 3]
    engine.backend.ref_text = "test reference"
    engine.backend.sample_rate = 24_000
    engine.backend.loaded = True
    return engine


def _audio() -> np.ndarray:
    samples = 24_000
    return (0.08 * np.sin(np.linspace(0, 2 * np.pi * 440, samples))).astype(np.float32)


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

    engine.backend.tts.infer = infer
    with (
        patch.object(engine.backend, "_apply_generation_settings"),
        patch.object(engine.backend, "_chunk_wer", return_value=_ACCEPTED_QUALITY),
    ):
        result = engine.synthesize("hello world")

    assert result.timing is not None
    attempts = result.timing.chunks[0]["attempts_detail"]
    assert len(attempts) == 2
    assert attempts[0]["wer"] is None
    assert attempts[0]["t_infer"] >= 0.008


def test_f0_timing_includes_pitch_analysis():
    engine = _engine()
    engine.backend.tts.infer = MagicMock(return_value=_audio())

    def slow_f0(*args, **kwargs):
        time.sleep(0.01)
        return 440.0

    with (
        patch.object(engine.backend, "_apply_generation_settings"),
        patch.object(engine.backend, "_chunk_wer", return_value=_ACCEPTED_QUALITY),
        patch("tinytalk.backends.neutts.compute_f0_mean", side_effect=slow_f0),
    ):
        result = engine.synthesize("hello world")

    assert result.timing is not None
    assert result.timing.chunks[0]["t_f0"] >= 0.008


def test_generation_settings_restore_even_when_request_fails():
    engine = _engine(max_retries=1)
    engine.backend.tts.infer = MagicMock(side_effect=ValueError("no speech tokens"))

    with patch.object(engine.backend, "_apply_generation_settings") as apply_settings:
        with pytest.raises(RuntimeError, match="produced no speech tokens"):
            engine.synthesize("hello world")

    assert len(apply_settings.call_args_list) == 3
    assert apply_settings.call_args_list[-1].args == (engine.backend.tts,)
    assert apply_settings.call_args_list[-1].kwargs == {}


def test_f0_continuity_resets_between_requests():
    engine = _engine()
    engine.backend.settings = replace(engine.settings, max_chars_per_chunk=5)
    audio = _audio()

    with (
        patch.object(engine.backend, "_synthesize_chunk", side_effect=lambda text, index, count: (
            audio, {"index": index, "attempts": 1, "duration": 1.0, "status": "accepted"}, 0
        )),
        patch.object(engine.backend, "_apply_generation_settings"),
        patch("tinytalk.backends.neutts.compute_f0_mean", return_value=440.0),
        patch("tinytalk.backends.neutts.normalize_f0", return_value=audio) as normalize,
    ):
        engine.synthesize("one two")
        engine.synthesize("three")

    normalize.assert_called_once()
    assert normalize.call_args.args[2] == 440.0
