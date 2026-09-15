from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import numpy as np

from .config import Backend, Settings
from .quality import transcribe_chunk


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


_BACKEND_CLASSES: dict[Backend, tuple[str, str]] = {
    "neutts": ("tinytalk.backends.neutts", "NeuTTSEngine"),
    "tinytauk": ("tinytalk.backends.tinytauk", "TinyTAuKEngine"),
}


def _backend_class(backend: Backend):
    module_name, class_name = _BACKEND_CLASSES[backend]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # A missing TinyTalk module indicates a packaging/programming error, not
        # an omitted optional backend extra. Do not disguise that case.
        if exc.name and exc.name.startswith("tinytalk."):
            raise
        missing = exc.name or "an optional dependency"
        raise RuntimeError(
            f"TinyTalk backend {backend!r} is not installed: missing Python module {missing!r}. "
            f"Install it with `pip install 'tinytalk[{backend}]'` or, from a source checkout, "
            f"`pip install -e '.[{backend}]'`."
        ) from exc
    return getattr(module, class_name)


def create_engine(settings: Settings) -> SpeechEngine:
    # Backend ML stacks are intentionally isolated. Import only the selected
    # implementation so one service instance needs only its own dependencies.
    engine_class = _backend_class(settings.backend)
    return engine_class(settings)


class TinyTalkEngine:
    """Backward-compatible constructor that dispatches to ``settings.backend``."""

    def __new__(cls, settings: Settings):
        return create_engine(settings)
