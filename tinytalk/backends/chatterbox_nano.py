from __future__ import annotations

import time
from pathlib import Path
from typing import override

import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

from ..chunking import split_tagged_text
from ..config import ChatterboxVoice, Settings, VoiceConfig, load_voice_config
from ..engine import ChunkResult, SpeechBackend, SynthesisRequest
from ..timing import ChunkTiming


class ChatterboxNanoBackend(SpeechBackend[SynthesisRequest]):
    tts: ChatterboxTurboTTS | None
    voices: VoiceConfig[ChatterboxVoice] | None
    voice_conditionals: dict[str | None, Conditionals]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "ResembleAI/chatterbox-nano", load_voice_config(settings, ChatterboxVoice))
        self.tts = None
        self.voice_conditionals = {}

    @override
    def load(self) -> None:
        if self.loaded:
            return
        references = self._references()
        model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
        self.voice_conditionals = {
            name: self._prepare_conditionals(model, reference)
            for name, reference in references.items()
        }
        self.tts = model
        self.sample_rate = int(model.sr)
        self.loaded = True

    def _references(self) -> dict[str | None, Path | None]:
        if self.voices is not None:
            return {name: preset.ref_audio for name, preset in self.voices.voices.items()}
        reference = self.settings.chatterbox_nano_ref_audio
        if reference is not None and not reference.is_file():
            raise ValueError(f"Chatterbox Nano reference audio does not exist: {reference}")
        return {None: reference}

    @staticmethod
    def _prepare_conditionals(model: ChatterboxTurboTTS, reference: Path | None) -> Conditionals:
        if reference is not None:
            # Preserve native reference levels and avoid upstream dtype promotion.
            model.prepare_conditionals(str(reference), exaggeration=0.0, norm_loudness=False)
        if model.conds is None:
            raise RuntimeError("Chatterbox Nano has no prepared conditioning")
        return model.conds

    @override
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
        voice: str | None = None,
    ) -> SynthesisRequest:
        if instructions and instructions.strip():
            raise ValueError("Chatterbox Nano uses native tags in input, not instructions")
        if speed is not None and speed != 1.0:
            raise ValueError("Chatterbox Nano supports only speed=1.0")
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")
        self.tts.conds = self.voice_conditionals[voice]
        return SynthesisRequest(instructions=None, speed=None, voice=voice)

    @override
    def prepare_chunks(self, text: str, max_chars_per_chunk: int) -> list[str]:
        return split_tagged_text(text, max_chars_per_chunk)

    @override
    def synthesize_chunk(
        self,
        request: SynthesisRequest,
        chunk_text: str,
        index: int,
        num_chunks: int,
    ) -> ChunkResult:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")
        start = time.perf_counter()
        audio = self.tts.generate(chunk_text).detach().cpu().numpy().astype(np.float32).reshape(-1)
        timing = ChunkTiming(
            index=index,
            attempts=1,
            duration=len(audio) / self.sample_rate,
            t_infer=time.perf_counter() - start,
            status="accepted",
        )
        return audio, timing, 0
