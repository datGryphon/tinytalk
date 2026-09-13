import io
import shutil
import subprocess
import wave

import numpy as np

_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
}

_FFMPEG_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "opus": ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
}


def silence(sample_rate: int, ms: int, dtype: np.dtype | type) -> np.ndarray:
    return np.zeros(int(sample_rate * (ms / 1000.0)), dtype=dtype)


def trim_edge_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.001,
    keep_ms: int = 180,
    leading: bool = True,
    trailing: bool = True,
) -> np.ndarray:
    mono = np.asarray(audio).squeeze()
    levels = np.abs(_as_float(mono))
    voiced = np.flatnonzero(levels > threshold)
    if voiced.size == 0:
        return mono

    keep = int(sample_rate * (keep_ms / 1000.0))
    start = max(int(voiced[0]) - keep, 0) if leading else 0
    stop = min(int(voiced[-1]) + keep + 1, mono.size) if trailing else mono.size
    return mono[start:stop]


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(audio).squeeze()
    pcm16 = _to_pcm16(mono)
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return out.getvalue()


def encode_audio(wav_bytes: bytes, response_format: str) -> tuple[bytes, str]:
    """Return (encoded body, media type) for the requested response_format.

    `wav` is passed through unchanged. `mp3` and `opus` are transcoded via
    ffmpeg. The opus container is Ogg.
    """
    if response_format == "wav":
        return wav_bytes, _MEDIA_TYPES["wav"]
    if response_format not in _FFMPEG_ARGS:
        raise ValueError(f"unsupported response_format: {response_format!r}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH, required for non-wav response_format")

    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            *_FFMPEG_ARGS[response_format],
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=True,
    )

    return result.stdout, _MEDIA_TYPES[response_format]


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio
    scaled = _as_float(audio)
    return (np.clip(scaled, -1.0, 1.0) * 32767.0).astype(np.int16)


def _as_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        return audio.astype(np.float32) / max(abs(info.min), info.max)
    return audio.astype(np.float32)


def chunk_rms(audio: np.ndarray) -> float:
    """RMS amplitude of the array."""
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


def clip_fraction(audio: np.ndarray, ceiling: float = 0.98) -> float:
    """Fraction of samples at or beyond the ceiling (clipping indicator)."""
    return float(np.mean(np.abs(_as_float(audio)) >= ceiling))


def duration_ratio(
    audio: np.ndarray,
    expected_chars: int,
    sample_rate: int = 24000,
    chars_per_second: float = 14.0,
) -> float:
    """Actual duration / expected duration. 1.0 = perfect, >> 1 = stuck, << 1 = dropped."""
    if expected_chars <= 0:
        return 1.0
    expected_sec = expected_chars / chars_per_second
    actual_sec = len(np.asarray(audio)) / sample_rate
    return actual_sec / expected_sec


def chunk_confidence(audio: np.ndarray, expected_chars: int) -> float:
    """
    Returns a confidence score in [0, 1]. Higher = better quality chunk.
    Returns 0.0 for silence, penalises clipping and extreme duration ratios.
    """
    rms = chunk_rms(audio)
    if rms < 0.01:
        return 0.0
    cf = clip_fraction(audio)
    dr = duration_ratio(audio, expected_chars)
    dur_score = max(0.0, 1.0 - abs(dr - 1.0) / 2.0)
    clip_score = 1.0 - min(cf * 10.0, 1.0)
    return dur_score * clip_score


def edge_fade(audio: np.ndarray, fade_ms: float = 3.0, sample_rate: int = 24000) -> np.ndarray:
    fade_samples = max(int(fade_ms / 1000.0 * sample_rate), 1)
    if len(audio) <= 2 * fade_samples:
        return audio.copy()
    result = np.asarray(audio).copy()
    result[:fade_samples] = np.linspace(0, 1.0, fade_samples) * result[:fade_samples]
    result[-fade_samples:] = np.linspace(1.0, 0, fade_samples) * result[-fade_samples:]
    return result


def loudness_normalize(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    audio_f = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(audio_f**2)))
    if rms < 1e-6:
        return np.asarray(audio, dtype=np.float32)
    result = audio_f * (target_rms / rms)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def peak_limit(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    return np.clip(np.asarray(audio, dtype=np.float64), -ceiling, ceiling).astype(np.float32)
