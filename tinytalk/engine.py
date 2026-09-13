from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Settings
from .wer import transcribe_chunk


@dataclass(frozen=True)
class RequestTiming:
    """Backend timing payload for one synthesis request."""

    chunks: list[dict]
    wer_fallbacks: int = 0


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    chunks: list[str]
    timing: RequestTiming | None = None


class SpeechEngine(Protocol):
    loaded: bool
    model_name: str

    def load(self) -> None: ...

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult: ...


def create_engine(settings: Settings) -> SpeechEngine:
    # The backends currently have incompatible ML dependency stacks. Import
    # only the selected implementation so one service instance needs only its
    # own runtime dependencies.
    if settings.backend == "neutts":
        from .backends.neutts import NeuTTSEngine

        return NeuTTSEngine(settings)
    if settings.backend == "tinytauk":
        from .backends.tinytauk import TinyTAuKEngine

        return TinyTAuKEngine(settings)
    raise ValueError(f"unsupported synthesis backend: {settings.backend!r}")


class TinyTalkEngine:
    """Backward-compatible constructor for callers that expect NeuTTS."""

    def __new__(cls, settings: Settings):
        from .backends.neutts import NeuTTSEngine

        return NeuTTSEngine(settings)
