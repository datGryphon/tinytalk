"""Tests for the flat NeuTTS reroll loop.

The loop: holds temperature constant, escalates repeat_penalty per attempt,
scores each with _chunk_wer (WER), keeps the lowest-WER attempt, early-exits
when wer <= wer_threshold.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("neutts")

from tinytalk.backends.neutts import NeuTTSEngine
from tinytalk.config import Settings


def _make_engine(settings: Settings) -> NeuTTSEngine:
    """Build an engine without loading the model. Caller sets its test state."""
    engine = NeuTTSEngine(settings)
    engine.ref_codes = [1, 2, 3]
    engine.ref_text = "test reference"
    engine.sample_rate = 24_000
    return engine


def _good_audio() -> np.ndarray:
    """Non-silent audio at normal RMS and duration for 14 chars (1s)."""
    sr = 24_000
    dur = int(sr * (14 / 14.0))
    return (np.sin(np.linspace(0, 2 * np.pi * 440, dur)) * (0.08 / (1 / (2**0.5)))).astype(
        np.float32
    )


# ── repeat_penalty escalation ────────────────────────────────────────────────


class TestRepeatPenaltyEscalation:

    def test_escapes_with_step(self):
        """repeat_penalty_override = base + attempt * step for each attempt."""
        settings = Settings(
            max_retries=2,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="",
            wer_threshold=-0.1,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)

        def mock_infer(*args, **kwargs):
            return _good_audio()

        engine.tts = MagicMock()
        engine.tts.infer = mock_infer

        rp_values: list[float] = []

        def spy_apply(*args, **kwargs):
            rp_values.append(kwargs.get("repeat_penalty_override"))

        with patch.object(engine, "_apply_generation_settings", side_effect=spy_apply):
            engine.synthesize("hello world test")

        loop_rp = [v for v in rp_values if v is not None]
        assert loop_rp == pytest.approx([1.0, 1.1, 1.2])

    def test_temperature_not_passed(self):
        """_apply_generation_settings has no temperature_override param."""
        settings = Settings(
            max_retries=1,
            temperature=1.0,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.20,
            wer_endpoint="",
            wer_threshold=-0.1,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)
        engine.tts = MagicMock()
        engine.tts.infer = MagicMock(return_value=_good_audio())

        kwarg_keys: list[list[str]] = []

        def spy_apply(*args, **kwargs):
            kwarg_keys.append(list(kwargs.keys()))

        with patch.object(engine, "_apply_generation_settings", side_effect=spy_apply):
            engine.synthesize("hello world test")

        loop_calls = [keys for keys in kwarg_keys if keys]
        assert all(keys == ["repeat_penalty_override"] for keys in loop_calls)
        assert len(loop_calls) == 2


# ── best-of-N by WER ────────────────────────────────────────────────────────


class TestBestOfNByWer:

    def test_keeps_lowest_wer(self):
        settings = Settings(
            max_retries=2,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="",
            wer_threshold=0.0,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)

        audio_a = np.zeros(24_000, dtype=np.float32)
        audio_b = _good_audio()

        tracker = [0]

        def mock_infer(*args, **kwargs):
            idx = tracker[0]
            tracker[0] += 1
            return audio_a if idx <= 1 else audio_b

        engine.tts = MagicMock()
        engine.tts.infer = mock_infer

        def fake_wer(wav, text):
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            return 0.8 if rms < 0.01 else 0.1

        with patch.object(engine, "_chunk_wer", side_effect=fake_wer):
            result = engine.synthesize("hello world test")

        rms = float(np.sqrt(np.mean(result.audio.astype(np.float64) ** 2)))
        assert rms > 0.01


# ── early-exit ───────────────────────────────────────────────────────────────


class TestEarlyExit:

    def test_exits_on_first_attempt_below_threshold(self):
        settings = Settings(
            max_retries=3,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="",
            wer_threshold=0.25,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)
        engine.tts = MagicMock()
        engine.tts.infer = MagicMock(return_value=_good_audio())

        call_count = 0

        def fake_wer(wav, text):
            nonlocal call_count
            call_count += 1
            return 0.1 if call_count == 1 else 1.0

        with patch.object(engine, "_chunk_wer", side_effect=fake_wer):
            engine.synthesize("hello world test")

        assert call_count == 1
        engine.tts.infer.assert_called_once()


# ── exhaust retries ──────────────────────────────────────────────────────────


class TestExhaustRetries:

    def test_runs_all_attempts_and_keeps_min_wer(self):
        settings = Settings(
            max_retries=2,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="",
            wer_threshold=0.0,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)

        audios = [np.zeros(24_000, dtype=np.float32)] * 3
        audios[1] = _good_audio()
        call_count = 0

        def mock_infer(*args, **kwargs):
            nonlocal call_count
            audio = audios[call_count]
            call_count += 1
            return audio

        engine.tts = MagicMock()
        engine.tts.infer = mock_infer

        def fake_wer(wav, text):
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            return 1.0 if rms < 0.01 else 0.05

        with patch.object(engine, "_chunk_wer", side_effect=fake_wer):
            result = engine.synthesize("hello world test")

        assert call_count == 3
        rms = float(np.sqrt(np.mean(result.audio.astype(np.float64) ** 2)))
        assert rms > 0.01


# ── fail-open ────────────────────────────────────────────────────────────────


class TestFailOpen:

    def test_no_endpoint_uses_chunk_confidence(self):
        settings = Settings(
            max_retries=0,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="",
            wer_threshold=-0.1,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)
        engine.tts = MagicMock()
        engine.tts.infer = MagicMock(return_value=_good_audio())

        with patch.object(engine, "_chunk_wer") as mock_wer:
            mock_wer.return_value = 0.3
            result = engine.synthesize("hello world test")
            mock_wer.assert_called_once()
        assert len(result.audio) > 0

    def test_endpoint_exception_falls_back_to_confidence(self):
        settings = Settings(
            max_retries=0,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="http://transcribe.local",
            wer_threshold=-0.1,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)
        engine.tts = MagicMock()
        engine.tts.infer = MagicMock(return_value=_good_audio())

        with patch(
            "tinytalk.backends.neutts.transcribe_chunk",
            side_effect=ConnectionError("nope"),
        ):
            result = engine.synthesize("hello world test")

        assert len(result.audio) > 0

    def test_silent_chunk_fails_open_retries(self):
        settings = Settings(
            max_retries=1,
            repeat_penalty=1.0,
            repeat_penalty_reroll_step=0.10,
            wer_endpoint="",
            wer_threshold=-0.1,
            ref_codes=Path(__file__).parent / "voices" / "jo.pt",
            ref_text=Path(__file__).parent / "voices" / "jo.txt",
        )
        engine = _make_engine(settings)
        silent = np.zeros(24_000, dtype=np.float32)
        engine.tts = MagicMock()
        engine.tts.infer = MagicMock(return_value=silent)

        result = engine.synthesize("hello world test")
        assert engine.tts.infer.call_count == 2
        assert len(result.audio) > 0
