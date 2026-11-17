import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Backend = Literal["neutts", "tinytauk", "omnivoice", "silencio2", "chatterbox_nano"]


def _env_name(field: str) -> str:
    return f"TINYTALK_{field.upper()}"


class Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", alias_generator=_env_name, validate_by_name=True,
    )

    backend: Backend = "neutts"
    model: str = "neuphonic/neutts-nano-q4-gguf"
    codec: str = "neuphonic/neucodec-onnx-decoder-int8"
    backbone_device: str = "cpu"
    ref_codes: Path = Path("/var/lib/tinytalk/ref_codes.pt")
    ref_text: Path = Path("/var/lib/tinytalk/ref_text.txt")
    tinytauk_model: str = "tencent/AuK-Flash"
    tinytauk_qwen_model: str = "Qwen/Qwen2.5-Omni-3B"
    tinytauk_profile: Path | None = None
    tinytauk_chars_per_second: float = Field(default=14.0, gt=0)
    omnivoice_model: str = "k2-fsa/OmniVoice"
    omnivoice_device: str = "cpu"
    omnivoice_language: str = ""
    omnivoice_ref_audio: Path | None = None
    omnivoice_ref_text: Path | None = None
    silencio2_voice: str = "260"
    chatterbox_nano_ref_audio: Path | None = None
    host: str = "0.0.0.0"
    port: int = 9002
    max_chars_per_chunk: int = Field(default=180, gt=0)
    inter_chunk_silence_ms: int = 60
    temperature: float = 1.0
    repeat_penalty: float = 1.0
    max_retries: int = Field(default=2, ge=0)
    repeat_penalty_reroll_step: float = 0.10
    wer_endpoint: str = ""
    wer_threshold: float = 0.25
    watermark: bool = False

    @field_validator(
        "tinytauk_profile", "omnivoice_ref_audio", "omnivoice_ref_text",
        "chatterbox_nano_ref_audio", mode="before",
    )
    @classmethod
    def _optional_path(cls, value: str | Path | None) -> str | Path | None:
        return None if isinstance(value, str) and not value.strip() else value


def load_settings() -> Settings:
    # Empty numeric/boolean/backend variables retain their defaults, while
    # explicitly configured string and path values still reach the model.
    return Settings.model_validate({
        name: value
        for field in Settings.model_fields.values()
        if (name := field.alias) is not None
        and (value := os.getenv(name)) is not None
        and (value != "" or field.annotation in (str, Path))
    })
