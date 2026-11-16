from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from .audio import silence
from .config import Backend, Settings
from .quality import QualityResult, transcribe_chunk

type ChunkResult = tuple[np.ndarray, dict[str, object], int]


@dataclass(frozen=True)
class RequestTiming:
    """Backend timing payload for one synthesis request."""

    chunks: list[dict[str, object]]
    wer_fallbacks: int = 0


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    chunks: list[str]
    timing: RequestTiming | None = None


class SpeechEngine(ABC):
    settings: Settings
    model_name: str
    sample_rate: int
    loaded: bool

    def __init__(self, settings: Settings, model_name: str) -> None:
        self.settings = settings
        self.model_name = model_name
        self.sample_rate = 24_000
        self.loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult: ...

    def _synthesize_chunks(
        self,
        chunks: list[str],
        synthesize_chunk: Callable[[str, int, int], ChunkResult],
    ) -> SynthesisResult:
        """Synthesize pre-split chunks and assemble native audio and request timings."""
        parts: list[np.ndarray] = []
        chunk_timings: list[dict[str, object]] = []
        wer_fallbacks = 0
        silence_ms = self.settings.inter_chunk_silence_ms

        for index, chunk in enumerate(chunks):
            wav, chunk_timing, chunk_fallbacks = synthesize_chunk(chunk, index, len(chunks))
            wer_fallbacks += chunk_fallbacks
            if index > 0 and silence_ms > 0:
                parts.append(silence(self.sample_rate, silence_ms, wav.dtype))
            parts.append(wav)
            chunk_timings.append(chunk_timing)

        return SynthesisResult(
            audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
            sample_rate=self.sample_rate,
            chunks=chunks,
            timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=wer_fallbacks),
        )

    @staticmethod
    def _quality_timing_fields(quality: QualityResult | None) -> dict[str, object]:
        if quality is None:
            return {
                "wer": None,
                "cer": None,
                "substitutions": None,
                "deletions": None,
                "insertions": None,
                "prompt_leak": False,
                "wer_source": None,
                "wer_fallback": False,
            }
        return quality.timing_fields()


_BACKEND_CLASSES: dict[Backend, tuple[str, str]] = {
    "neutts": ("tinytalk.backends.neutts", "NeuTTSEngine"),
    "tinytauk": ("tinytalk.backends.tinytauk", "TinyTAuKEngine"),
    "omnivoice": ("tinytalk.backends.omnivoice", "OmniVoiceEngine"),
    "chatterbox_nano": ("tinytalk.backends.chatterbox_nano", "ChatterboxNanoEngine"),
    "silencio2": ("tinytalk.backends.silencio2", "Silencio2Engine"),
}


def _backend_class(backend: Backend) -> type[SpeechEngine]:
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
