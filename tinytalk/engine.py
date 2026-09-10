from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    chunks: list[str]


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
    # NeuTTS and TinyTAuK currently require incompatible Torch/Transformers
    # stacks, so import only the backend selected for this service instance.
    if settings.backend == "neutts":
        from .backends.neutts import NeuTTSEngine

        return NeuTTSEngine(settings)
    if settings.backend == "tinytauk":
        from .backends.tinytauk import TinyTAuKEngine

        return TinyTAuKEngine(settings)
    raise ValueError(f"unsupported synthesis backend: {settings.backend!r}")
