import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = "neuphonic/neutts-nano-q4-gguf"
    codec: str = "neuphonic/neucodec-onnx-decoder-int8"
    backbone_device: str = "cpu"
    ref_codes: Path = Path("/var/lib/tinytalk/ref_codes.pt")
    ref_text: Path = Path("/var/lib/tinytalk/ref_text.txt")
    host: str = "0.0.0.0"
    port: int = 9002
    max_chars_per_chunk: int = 180
    inter_chunk_silence_ms: int = 60
    temperature: float = 1.0
    repeat_penalty: float = 1.0
    max_retries: int = 2
    repeat_penalty_reroll_step: float = 0.10
    wer_endpoint: str = ""
    wer_threshold: float = 0.25
    watermark: bool = False


def load_settings() -> Settings:
    return Settings(
        model=os.getenv("TINYTALK_MODEL", Settings.model),
        codec=os.getenv("TINYTALK_CODEC", Settings.codec),
        backbone_device=os.getenv("TINYTALK_BACKBONE_DEVICE", Settings.backbone_device),
        ref_codes=Path(os.getenv("TINYTALK_REF_CODES", str(Settings.ref_codes))),
        ref_text=Path(os.getenv("TINYTALK_REF_TEXT", str(Settings.ref_text))),
        host=os.getenv("TINYTALK_HOST", Settings.host),
        port=_int_env("TINYTALK_PORT", Settings.port),
        max_chars_per_chunk=_int_env(
            "TINYTALK_MAX_CHARS_PER_CHUNK", Settings.max_chars_per_chunk
        ),
        inter_chunk_silence_ms=_int_env(
            "TINYTALK_INTER_CHUNK_SILENCE_MS", Settings.inter_chunk_silence_ms
        ),
        temperature=_float_env("TINYTALK_TEMPERATURE", Settings.temperature),
        repeat_penalty=_float_env("TINYTALK_REPEAT_PENALTY", Settings.repeat_penalty),
        max_retries=_int_env("TINYTALK_MAX_RETRIES", Settings.max_retries),
        repeat_penalty_reroll_step=_float_env(
            "TINYTALK_REPEAT_PENALTY_REROLL_STEP", Settings.repeat_penalty_reroll_step
        ),
        wer_endpoint=os.getenv("TINYTALK_WER_ENDPOINT", Settings.wer_endpoint),
        wer_threshold=_float_env("TINYTALK_WER_THRESHOLD", Settings.wer_threshold),
        watermark=_bool_env("TINYTALK_WATERMARK", Settings.watermark),
    )


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    low = raw.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")
