from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from .audio import silence
from .chunking import split_text
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


@dataclass
class SynthesisRequest:
    instructions: str | None
    speed: float | None


class SpeechBackend[RequestT: SynthesisRequest](ABC):
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
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
    ) -> RequestT: ...

    def prepare_chunks(self, text: str) -> list[str]:
        return split_text(text, self.settings.max_chars_per_chunk)

    @abstractmethod
    def synthesize_chunk(
        self,
        request: RequestT,
        chunk_text: str,
        index: int,
        num_chunks: int,
    ) -> ChunkResult: ...

    def finalize_request(self, request: RequestT) -> None:
        pass

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


class SpeechEngine[BackendT: SpeechBackend[Any]]:
    backend: BackendT

    def __init__(self, backend: BackendT) -> None:
        self.backend = backend

    @property
    def settings(self) -> Settings:
        return self.backend.settings

    @property
    def loaded(self) -> bool:
        return self.backend.loaded

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    def load(self) -> None:
        self.backend.load()

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        if not self.loaded:
            raise RuntimeError("engine not loaded. Call load() first")
        request = self.backend.prepare_request(instructions=instructions, speed=speed)
        try:
            chunks = self.backend.prepare_chunks(text)
            if not chunks:
                raise ValueError("cannot synthesize empty text")
            parts: list[np.ndarray] = []
            chunk_timings: list[dict[str, object]] = []
            wer_fallbacks = 0
            silence_ms = self.settings.inter_chunk_silence_ms

            for index, chunk in enumerate(chunks):
                wav, timing, fallbacks = self.backend.synthesize_chunk(
                    request, chunk, index, len(chunks)
                )
                wer_fallbacks += fallbacks
                if index > 0 and silence_ms > 0:
                    parts.append(silence(self.sample_rate, silence_ms, wav.dtype))
                parts.append(wav)
                chunk_timings.append(timing)

            return SynthesisResult(
                audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
                sample_rate=self.sample_rate,
                chunks=chunks,
                timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=wer_fallbacks),
            )
        finally:
            self.backend.finalize_request(request)


_BACKEND_CLASSES: dict[Backend, tuple[str, str]] = {
    "neutts": ("tinytalk.backends.neutts", "NeuTTSBackend"),
    "tinytauk": ("tinytalk.backends.tinytauk", "TinyTAuKBackend"),
    "omnivoice": ("tinytalk.backends.omnivoice", "OmniVoiceBackend"),
    "chatterbox_nano": ("tinytalk.backends.chatterbox_nano", "ChatterboxNanoBackend"),
    "silencio2": ("tinytalk.backends.silencio2", "Silencio2Backend"),
}


def _backend_class(backend: Backend) -> type[SpeechBackend[Any]]:
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


def create_engine(settings: Settings) -> SpeechEngine[SpeechBackend[Any]]:
    # Backend ML stacks are intentionally imported lazily. This preserves clean
    # implementation boundaries even when multiple backends eventually share a
    # qualified Python/Torch environment.
    backend_class = _backend_class(settings.backend)
    return SpeechEngine(backend_class(settings))


TinyTalkEngine = create_engine
