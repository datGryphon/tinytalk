import wave
from io import BytesIO

import numpy as np

from tinytalk.audio import silence, to_wav_bytes, trim_edge_silence


def test_silence_length_uses_sample_rate_and_ms():
    assert len(silence(24_000, 100, np.float32)) == 2400


def test_to_wav_bytes_writes_mono_pcm16():
    wav_bytes = to_wav_bytes(np.zeros(2400, dtype=np.float32), 24_000)
    with wave.open(BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24_000
        assert wav.getnframes() == 2400


def test_trim_edge_silence_removes_only_outer_silence():
    audio = np.concatenate(
        [
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 0.25,
            np.zeros(500, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 0.25,
            np.zeros(1000, dtype=np.float32),
        ]
    )

    trimmed = trim_edge_silence(audio, 1000, keep_ms=10)

    assert len(trimmed) == 720
    assert np.count_nonzero(trimmed[:10]) == 0
    assert np.count_nonzero(trimmed[-10:]) == 0
    assert len(trimmed[110:610]) == 500


def test_trim_edge_silence_can_preserve_outer_edges():
    audio = np.concatenate(
        [
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 0.25,
            np.zeros(1000, dtype=np.float32),
        ]
    )

    trimmed = trim_edge_silence(audio, 1000, keep_ms=10, leading=False, trailing=False)

    assert len(trimmed) == len(audio)
