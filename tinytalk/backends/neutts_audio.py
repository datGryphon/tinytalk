from __future__ import annotations

import librosa
import numpy as np
import parselmouth
import parselmouth.praat

from ..audio import _as_float

_MAX_F0_SCALE = 2 ** (2 / 12)  # ~2 semitones


def _pyin(audio_f: np.ndarray, sample_rate: int):
    return librosa.pyin(
        audio_f,
        fmin=50.0,
        fmax=800.0,
        sr=sample_rate,
        frame_length=1024,
        hop_length=256,
    )


def compute_f0_mean(
    audio: np.ndarray,
    sample_rate: int,
) -> float:
    """Compute mean F0 of a chunk (fast, no reconstruction needed)."""
    if len(audio) < int(0.1 * sample_rate):
        return 0.0
    audio_f = _as_float(np.asarray(audio).squeeze()).astype(np.float64)
    f0, voiced_flag, _ = _pyin(audio_f, sample_rate)
    voiced = f0[voiced_flag]
    return float(voiced.mean()) if voiced.size > 0 else 0.0


def normalize_f0(
    audio: np.ndarray,
    sample_rate: int,
    ref_f0_mean: float,
) -> np.ndarray:
    """Normalize chunk F0 to match reference mean pitch using parselmouth PSOLA."""
    if len(audio) < int(0.1 * sample_rate):
        return audio

    audio_f = _as_float(np.asarray(audio).squeeze()).astype(np.float64)

    f0, voiced_flag, _ = _pyin(audio_f, sample_rate)
    voiced_frames = f0[voiced_flag]
    if voiced_frames.size == 0:
        return audio

    current_mean = voiced_frames.mean()
    if current_mean < 1.0:
        return audio

    if abs(current_mean - ref_f0_mean) / ref_f0_mean < 0.05:
        return audio

    scale = ref_f0_mean / current_mean
    scale = min(max(scale, 1 / _MAX_F0_SCALE), _MAX_F0_SCALE)
    snd = parselmouth.Sound(audio_f, sampling_frequency=sample_rate)
    pitch_floor = float(f0[f0 > 0].min()) if (f0 > 0).any() else 50.0
    pitch_ceil = float(f0[f0 > 0].max()) if (f0 > 0).any() else 800.0

    manipulation = parselmouth.praat.call(snd, "To Manipulation", 0.01, pitch_floor, pitch_ceil)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call(pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, scale)
    parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
    resynth = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")

    return np.asarray(resynth.values, dtype=np.float32).flatten()
