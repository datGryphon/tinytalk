import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Backend = Literal["neutts", "tinytauk", "omnivoice", "silencio2", "chatterbox_nano"]


def _env_name(field: str) -> str:
    return f"TINYTALK_{field.upper()}"


class VoicePreset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ref_codes: Path | None = None
    ref_text: Path | None = None
    ref_audio: Path | None = None
    voice_id: str | None = None
    description: str | None = None
    max_chars_per_chunk: int | None = Field(default=None, gt=0)
    inter_chunk_silence_ms: int | None = Field(default=None, ge=0)


class VoiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend
    default: str
    voices: dict[str, VoicePreset]

    @model_validator(mode="after")
    def validate_voices(self) -> "VoiceConfig":
        if self.default not in self.voices:
            raise ValueError("default voice must be present in voices")
        required = {
            "neutts": ("ref_codes", "ref_text"),
            "tinytauk": (),
            "omnivoice": ("ref_audio", "ref_text"),
            "silencio2": ("voice_id",),
            "chatterbox_nano": ("ref_audio",),
        }[self.backend]
        allowed = set(required) | ({"description"} if self.backend == "tinytauk" else set())

        for name, preset in self.voices.items():
            if not name.strip():
                raise ValueError("voice names cannot be empty")
            missing = [field for field in required if getattr(preset, field) is None]
            if missing:
                raise ValueError(f"voice {name!r} requires {', '.join(required)}")
            unsupported = [
                field for field in ("ref_codes", "ref_text", "ref_audio", "voice_id", "description")
                if field not in allowed and getattr(preset, field) is not None
            ]
            if unsupported:
                raise ValueError(f"voice {name!r} has unsupported fields for {self.backend}")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", alias_generator=_env_name, validate_by_name=True,
    )

    backend: Backend = "neutts"
    voices_file: Path | None = None
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
        "voices_file", "tinytauk_profile", "omnivoice_ref_audio", "omnivoice_ref_text",
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


def load_voice_config(settings: Settings) -> VoiceConfig | None:
    if settings.voices_file is None:
        return None
    data = json.loads(settings.voices_file.read_text(encoding="utf-8"))
    config = VoiceConfig.model_validate(data)
    if config.backend != settings.backend:
        raise ValueError("voice configuration backend does not match TINYTALK_BACKEND")
    return config
