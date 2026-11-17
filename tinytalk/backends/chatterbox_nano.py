from __future__ import annotations

import time
from pathlib import Path
from typing import override

import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

from ..chunking import split_tagged_text
from ..config import Settings
from ..engine import ChunkResult, SpeechBackend, SynthesisRequest
from ..timing import ChunkTiming


class ChatterboxNanoBackend(SpeechBackend[SynthesisRequest]):
    tts: ChatterboxTurboTTS | None
    default_conditionals: Conditionals | None
    voice_conditionals: dict[Path, Conditionals]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "ResembleAI/chatterbox-nano")
        self.tts = None
        self.default_conditionals = None
        self.voice_conditionals = {}

    @override
    def load(self) -> None:
        if self.loaded:
            return
        reference = self.settings.chatterbox_nano_ref_audio
        if reference is not None and not reference.is_file():
            raise ValueError("Chatterbox Nano reference audio does not exist")
        model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
        if reference is not None:
            # Preserve reference level and avoid upstream's normalization dtype promotion.
            model.prepare_conditionals(str(reference), exaggeration=0.0, norm_loudness=False)
        self.default_conditionals = model.conds
        if self.settings.voices_file is not None:
            from ..config import load_voice_config

            voices = load_voice_config(self.settings)
            if voices is not None:
                for preset in voices.voices.values():
                    if preset.ref_audio is None or not preset.ref_audio.is_file():
                        raise ValueError("Chatterbox Nano named reference audio does not exist")
                    model.prepare_conditionals(
                        str(preset.ref_audio), exaggeration=0.0, norm_loudness=False
                    )
                    if model.conds is None:
                        raise RuntimeError("Chatterbox Nano did not prepare conditioning")
                    self.voice_conditionals[preset.ref_audio] = model.conds
            model.conds = self.default_conditionals
        self.tts = model
        self.sample_rate = int(model.sr)
        self.loaded = True

    @override
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
    ) -> SynthesisRequest:
        if instructions and instructions.strip():
            raise ValueError("Chatterbox Nano uses native tags in input, not instructions")
        if speed is not None and speed != 1.0:
            raise ValueError("Chatterbox Nano supports only speed=1.0")
        return SynthesisRequest(instructions=None, speed=None)

    @override
    def prepare_chunks(self, text: str, max_chars_per_chunk: int | None = None) -> list[str]:
        return split_tagged_text(text, max_chars_per_chunk or self.settings.max_chars_per_chunk)

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
        if request.voice is not None and request.voice.ref_audio is not None:
            self.tts.conds = self.voice_conditionals[request.voice.ref_audio]
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

    @override
    def finalize_request(self, request: SynthesisRequest) -> None:
        if self.tts is not None and self.settings.voices_file is not None:
            self.tts.conds = self.default_conditionals
