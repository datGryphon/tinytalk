"""Tests for deterministic audio post-processing helpers."""

import numpy as np
import pytest

from tinytalk.audio import (
    chunk_confidence,
    chunk_rms,
    clip_fraction,
    duration_ratio,
    edge_fade,
    loudness_normalize,
    peak_limit,
)


class TestChunkRms:

    def test_zero_signal(self):
        assert chunk_rms(np.zeros(1000, dtype=np.float32)) == pytest.approx(0.0)

    def test_known_rms(self):
        samples = np.sin(np.linspace(0, 100 * 2 * np.pi, 1000))
        expected = 1.0 / (2**0.5)
        assert chunk_rms(samples) == pytest.approx(expected, abs=1e-3)

    def test_constant(self):
        assert chunk_rms(np.full(500, 0.5, dtype=np.float32)) == pytest.approx(0.5)


class TestClipFraction:

    def test_no_clipping(self):
        arr = np.linspace(-0.97, 0.97, 1000)
        assert clip_fraction(arr) == pytest.approx(0.0)

    def test_all_clipped(self):
        arr = np.ones(500) * 0.99
        assert clip_fraction(arr) == pytest.approx(1.0)

    def test_half_clipped(self):
        arr = np.concatenate([np.zeros(500), np.full(500, 1.0)])
        assert clip_fraction(arr) == pytest.approx(0.5)


class TestDurationRatio:

    def test_nominal(self):
        arr = np.sin(np.linspace(0, 2 * np.pi * 440, 24000))
        assert duration_ratio(arr, 14) == pytest.approx(1.0, abs=1e-6)

    def test_too_short(self):
        arr = np.sin(np.linspace(0, np.pi * 440, 12000))
        assert duration_ratio(arr, 14) == pytest.approx(0.5)

    def test_too_long(self):
        arr = np.sin(np.linspace(0, 6 * np.pi * 440, 72000))
        assert duration_ratio(arr, 14) == pytest.approx(3.0)

    def test_zero_chars(self):
        arr = np.zeros(100, dtype=np.float32)
        assert duration_ratio(arr, 0) == pytest.approx(1.0)


class TestChunkConfidence:

    def test_silent(self):
        assert chunk_confidence(np.zeros(1000, dtype=np.float32), 14) == pytest.approx(0.0)

    def test_good_chunk(self):
        sr = 24000
        duration = int(sr * (14 / 14.0))
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, duration)) * (0.08 / (1 / (2**0.5)))
        score = chunk_confidence(samples, 14)
        assert score > 0.5
        assert score <= 1.0

    def test_clipped_chunk(self):
        arr = np.full(1000, 0.99, dtype=np.float32)
        score = chunk_confidence(arr, 14)
        assert score == pytest.approx(0.0)

    def test_too_short_chunk(self):
        arr = np.sin(np.linspace(0, np.pi * 440, 4800)) * 0.08
        score = chunk_confidence(arr, 14)
        assert score > 0.0
        assert score < 1.0


def test_loudness_normalize_scales_up():
    np.random.seed(42)
    raw = np.random.randn(24_000).astype(np.float32) * 0.01
    result = loudness_normalize(raw, target_rms=0.08)
    rms = float(np.sqrt(np.mean(result.astype(np.float64)**2)))
    assert abs(rms - 0.08) < 0.005
    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_loudness_normalize_silent():
    raw = np.zeros(24_000, dtype=np.float32)
    result = loudness_normalize(raw, target_rms=0.08)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    assert np.all(result == 0.0)


def test_peak_limit_clips():
    raw = np.array([-1.2, -0.5, 0.0, 0.5, 1.2], dtype=np.float32)
    result = peak_limit(raw, ceiling=0.98)
    assert np.all(np.abs(result) <= 0.98)
    assert result[0] == -0.98
    assert result[4] == 0.98


def test_edge_fade_tapers():
    raw = np.full(24_000, 0.5, dtype=np.float32)
    result = edge_fade(raw, fade_ms=3.0, sample_rate=24_000)
    fade_samples = 72
    assert result[0] < 0.01
    assert result[-1] < 0.01
    middle = result[fade_samples + 10 : len(result) - fade_samples - 10]
    assert np.allclose(middle, 0.5, atol=0.01)


def test_edge_fade_too_short():
    raw = np.full(50, 0.5, dtype=np.float32)
    result = edge_fade(raw, fade_ms=3000.0, sample_rate=24_000)
    np.testing.assert_array_equal(result, raw)
