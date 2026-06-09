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
