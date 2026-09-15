from __future__ import annotations

import numpy as np
import pytest

from tinytalk.quality import (
    QualityResult,
    char_error_rate,
    evaluate_audio,
    evaluate_transcript,
    quality_is_acceptable,
    word_error_rate,
)


def _audio() -> np.ndarray:
    return (0.08 * np.sin(np.linspace(0, 2 * np.pi * 440, 24_000))).astype(np.float32)


def test_word_error_rate_retains_edit_breakdown():
    score = word_error_rate("hello world", "hello extra world")

    assert score.rate == pytest.approx(0.5)
    assert score.substitutions == 0
    assert score.deletions == 0
    assert score.insertions == 1
    assert score.reference_units == 2


def test_word_error_rate_normalizes_case_and_punctuation():
    score = word_error_rate("Hello, WORLD!", "hello world")
    assert score.rate == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("don't stop", "dont stop"),
        ("don’t stop", "dont stop"),
        ("weʼre ready", "were ready"),
        ("it＇s done", "its done"),
    ],
)
def test_word_error_rate_normalizes_spoken_word_apostrophes(reference, hypothesis):
    score = word_error_rate(reference, hypothesis)
    assert score.rate == pytest.approx(0.0)


def test_char_error_rate_ignores_spacing_after_normalization():
    score = char_error_rate("hello world", "helloworld")
    assert score.rate == pytest.approx(0.0)


def test_char_error_rate_normalizes_apostrophe_variants():
    score = char_error_rate("don’t", "dont")
    assert score.rate == pytest.approx(0.0)


def test_prompt_leak_requires_aligned_unexpected_words_from_prompt():
    quality = evaluate_transcript(
        "The deployment completed successfully.",
        "Speak calmly and deliberately. The deployment completed successfully.",
        prompt_text="Speak calmly and deliberately.",
    )

    assert quality.prompt_leak is True
    assert quality.insertions == 4
    assert quality.wer > 0.0


def test_prompt_leak_detects_prompt_words_that_replace_target_prefix():
    quality = evaluate_transcript(
        "The deployment completed successfully, although it took considerably longer than expected.",
        "Speak in a tired but relieved, compactly et cetera, although it took considerably longer than expected.",
        prompt_text="Speak in a tired but relieved technical narrator voice.",
    )

    assert quality.prompt_leak is True
    assert quality.substitutions is not None
    assert quality.substitutions > 0


def test_unrelated_insertion_is_not_classified_as_prompt_leak():
    quality = evaluate_transcript(
        "The deployment completed successfully.",
        "Well, the deployment completed successfully.",
        prompt_text="Speak calmly and deliberately.",
    )

    assert quality.prompt_leak is False
    assert quality.insertions == 1


def test_scaffold_leak_is_detected_without_instruction_overlap():
    quality = evaluate_transcript(
        "The deployment completed successfully.",
        "The content to speak is the deployment completed successfully.",
        scaffold_text="Generate speech based on the following description. The content to speak is.",
    )

    assert quality.prompt_leak is True


def test_acceptance_uses_wer_cer_and_prompt_leak():
    good = QualityResult(
        wer=0.1,
        cer=0.05,
        substitutions=0,
        deletions=0,
        insertions=1,
        prompt_leak=False,
        source="whisper",
        fallback=False,
    )
    bad_cer = QualityResult(**{**good.__dict__, "cer": 0.4})
    leaked = QualityResult(**{**good.__dict__, "prompt_leak": True})

    assert quality_is_acceptable(good, 0.25) is True
    assert quality_is_acceptable(bad_cer, 0.25) is False
    assert quality_is_acceptable(leaked, 0.25) is False


def test_evaluate_audio_uses_transcription_when_endpoint_is_configured():
    quality = evaluate_audio(
        _audio(),
        24_000,
        target_text="hello world",
        endpoint="http://asr.local",
        transcribe=lambda *_: "hello world",
    )

    assert quality.source == "whisper"
    assert quality.fallback is False
    assert quality.wer == pytest.approx(0.0)
    assert quality.cer == pytest.approx(0.0)


def test_evaluate_audio_falls_back_without_exposing_transcript():
    def fail(*_args):
        raise ConnectionError("offline")

    quality = evaluate_audio(
        _audio(),
        24_000,
        target_text="hello world",
        endpoint="http://asr.local",
        transcribe=fail,
    )

    assert quality.source == "confidence"
    assert quality.fallback is True
    assert quality.cer is None
    assert "transcript" not in quality.timing_fields()
