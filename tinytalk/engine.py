from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .audio import AudioSamples, silence
from .chunking import split_text
from .config import NamedVoices, Settings, VoicePreset
from .quality import transcribe_chunk
from .timing import ChunkTiming, RequestTiming

if TYPE_CHECKING:
    from .backends.neutts import NeuTTSRequest

type ChunkResult = tuple[AudioSamples, ChunkTiming, int]


class SynthesisResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    audio: AudioSamples
    sample_rate: int
    chunks: list[str]
    timing: RequestTiming | None = None

    @field_validator("audio")
    @classmethod
    def _validate_audio(cls, audio: AudioSamples) -> AudioSamples:
        if audio.dtype != np.float32:
            raise ValueError("synthesis audio must be float32")
        return audio


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    instructions: str | None
    speed: float | None
    voice: str | None = None


class UnknownVoiceError(ValueError):
    pass


class SpeechBackend[RequestT: SynthesisRequest](ABC):
    settings: Settings
    model_name: str
    sample_rate: int
    loaded: bool
    voices: NamedVoices | None

    def __init__(self, settings: Settings, model_name: str, voices: NamedVoices | None = None) -> None:
        self.settings = settings
        self.model_name = model_name
        self.sample_rate = 24_000
        self.loaded = False
        self.voices = voices

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
        voice: str | None = None,
    ) -> RequestT: ...

    def prepare_chunks(self, text: str, max_chars_per_chunk: int) -> list[str]:
        return split_text(text, max_chars_per_chunk)

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


class SpeechEngine[RequestT: SynthesisRequest]:
    backend: SpeechBackend[RequestT]

    def __init__(self, backend: SpeechBackend[RequestT]) -> None:
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

    def resolve_voice(self, name: str | None) -> tuple[str | None, VoicePreset]:
        config = self.backend.voices
        if config is None:
            return None, VoicePreset()
        selected = config.default if name is None else name
        if selected not in config.voices:
            raise UnknownVoiceError(f"unknown voice: {selected}")
        return selected, config.voices[selected]

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
        voice: str | None = None,
    ) -> SynthesisResult:
        if not self.loaded:
            raise RuntimeError("engine not loaded. Call load() first")
        name, preset = self.resolve_voice(voice)
        max_chars = preset.max_chars_per_chunk or self.settings.max_chars_per_chunk
        silence_ms = preset.inter_chunk_silence_ms
        if silence_ms is None:
            silence_ms = self.settings.inter_chunk_silence_ms
        request = self.backend.prepare_request(instructions=instructions, speed=speed, voice=name)
        try:
            chunks = self.backend.prepare_chunks(text, max_chars)
            if not chunks:
                raise ValueError("cannot synthesize empty text")
            parts: list[AudioSamples] = []
            chunk_timings: list[ChunkTiming] = []
            wer_fallbacks = 0

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


def create_engine(
    settings: Settings,
) -> SpeechEngine[SynthesisRequest] | SpeechEngine[NeuTTSRequest]:
    # Lazy imports allow a minimal service installation with only its backend extra.
    backend = settings.backend
    try:
        match backend:
            case "neutts":
                from .backends.neutts import NeuTTSBackend

                return SpeechEngine(NeuTTSBackend(settings))
            case "tinytauk":
                from .backends.tinytauk import TinyTAuKBackend

                return SpeechEngine(TinyTAuKBackend(settings))
            case "omnivoice":
                from .backends.omnivoice import OmniVoiceBackend

                return SpeechEngine(OmniVoiceBackend(settings))
            case "chatterbox_nano":
                from .backends.chatterbox_nano import ChatterboxNanoBackend

                return SpeechEngine(ChatterboxNanoBackend(settings))
            case "silencio2":
                from .backends.silencio2 import Silencio2Backend

                return SpeechEngine(Silencio2Backend(settings))
            case _:
                raise ValueError(f"unsupported synthesis backend: {backend!r}")
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


TinyTalkEngine = create_engine
