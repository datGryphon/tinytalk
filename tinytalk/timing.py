from typing import Literal, NotRequired, TypedDict


type AttemptStatus = Literal["accepted", "fallback"]
type VoiceMode = Literal["auto", "clone", "design"]
type QualitySource = Literal["whisper", "confidence"]


class QualityTiming(TypedDict, total=False):
    # Unscored backends omit these fields; failed attempts use null scores.
    wer: float | None
    cer: float | None
    substitutions: int | None
    deletions: int | None
    insertions: int | None
    prompt_leak: bool
    wer_source: QualitySource | None
    wer_fallback: bool


class AttemptTiming(QualityTiming):
    attempt: int
    t_infer: float
    t_dsp: float
    t_wer_check: float
    repeat_penalty: float | None
    status: AttemptStatus | None
    seed: NotRequired[int]
    gen_seconds: NotRequired[float]
    class_temperature: NotRequired[float]
    mode: NotRequired[VoiceMode]


class ChunkTiming(QualityTiming):
    index: int
    attempts: int
    status: AttemptStatus
    duration: float
    t_infer: NotRequired[float]
    t_f0: NotRequired[float]
    repeat_penalty: NotRequired[float | None]
    seed: NotRequired[int | None]
    gen_seconds: NotRequired[float]
    class_temperature: NotRequired[float]
    mode: NotRequired[VoiceMode]
    attempts_detail: NotRequired[list[AttemptTiming]]
