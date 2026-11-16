from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import numpy as np

from .audio import silence
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


def synthesize_chunks(
    chunks: list[str],
    sample_rate: int,
    inter_chunk_silence_ms: int,
    synthesize_chunk: Callable[[str, int, int], tuple[np.ndarray, dict, int]],
) -> SynthesisResult:
    """Synthesize pre-split chunks and assemble native audio and request timings."""
    parts: list[np.ndarray] = []
    chunk_timings: list[dict] = []
    wer_fallbacks = 0

    for index, chunk in enumerate(chunks):
        wav, chunk_timing, chunk_fallbacks = synthesize_chunk(chunk, index, len(chunks))
        wer_fallbacks += chunk_fallbacks
        if index > 0 and inter_chunk_silence_ms > 0:
            parts.append(silence(sample_rate, inter_chunk_silence_ms, wav.dtype))
        parts.append(wav)
        chunk_timings.append(chunk_timing)

    return SynthesisResult(
        audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
        sample_rate=sample_rate,
        chunks=chunks,
        timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=wer_fallbacks),
    )


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
    "omnivoice": ("tinytalk.backends.omnivoice", "OmniVoiceEngine"),
    "silencio2": ("tinytalk.backends.silencio2", "Silencio2Engine"),
}


def _backend_class(backend: Backend):
    try:
        module_name, class_name = _BACKEND_CLASSES[backend]
    except KeyError:
        raise ValueError(f"unsupported synthesis backend: {backend!r}") from None

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
            f"From a source checkout run `uv sync --frozen --extra {backend}`."
        ) from exc
    return getattr(module, class_name)


def create_engine(settings: Settings) -> SpeechEngine:
    # Backend ML stacks are intentionally imported lazily. This preserves clean
    # implementation boundaries even when multiple backends eventually share a
    # qualified Python/Torch environment.
    engine_class = _backend_class(settings.backend)
    return engine_class(settings)


class TinyTalkEngine:
    """Backward-compatible constructor that dispatches to ``settings.backend``."""

    def __new__(cls, settings: Settings):
        return create_engine(settings)
