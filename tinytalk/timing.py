from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import Backend


type AttemptStatus = Literal["accepted", "fallback"]
type VoiceMode = Literal["auto", "clone", "design"]
type QualitySource = Literal["whisper", "confidence"]


class QualityTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Unscored backends omit these fields from telemetry. Failed scored attempts
    # explicitly emit null scores, preserving the existing log contract.
    wer: float | None = None
    cer: float | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None
    prompt_leak: bool = False
    wer_source: QualitySource | None = None
    wer_fallback: bool = False


class AttemptTiming(QualityTiming):
    attempt: int
    t_infer: float
    t_dsp: float
    t_wer_check: float
    repeat_penalty: float | None
    status: AttemptStatus | None = None
    seed: int | None = None
    gen_seconds: float | None = None
    class_temperature: float | None = None
    mode: VoiceMode | None = None


class ChunkTiming(QualityTiming):
    index: int
    attempts: int
    status: AttemptStatus
    duration: float
    t_infer: float | None = None
    t_f0: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    gen_seconds: float | None = None
    class_temperature: float | None = None
    mode: VoiceMode | None = None
    attempts_detail: list[AttemptTiming] = Field(default_factory=list)


class RequestTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[ChunkTiming]
    wer_fallbacks: int = 0

    @property
    def total_attempts(self) -> int:
        return sum(chunk.attempts for chunk in self.chunks)


class RequestLog(BaseModel):
    elapsed: float
    backend: Backend
    chunks: int | None
    audio_seconds: float | None
    rtf: float | None
    attempts: int | None
    wer_fallbacks: int | None
    timing: list[ChunkTiming] | None
    error: str | None = None
