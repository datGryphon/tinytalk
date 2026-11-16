from __future__ import annotations

import time

import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS

from ..chunking import split_tagged_text
from ..config import Settings
from ..engine import SynthesisResult, synthesize_chunks


class ChatterboxNanoEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts = None
        self.loaded = False
        self.model_name = "ResembleAI/chatterbox-nano"
        self.sample_rate = 24000

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
        self.tts = model
        self.sample_rate = int(model.sr)
        self.loaded = True

    def synthesize(
        self, text: str, *, instructions: str | None = None, speed: float | None = None
    ) -> SynthesisResult:
        if not self.loaded:
            raise RuntimeError("engine not loaded. Call load() first")
        if instructions and instructions.strip():
            raise ValueError("Chatterbox Nano uses native tags in input, not instructions")
        if speed is not None and speed != 1.0:
            raise ValueError("Chatterbox Nano supports only speed=1.0")
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        chunks = split_tagged_text(text, self.settings.max_chars_per_chunk)
        return synthesize_chunks(
            chunks, self.sample_rate, self.settings.inter_chunk_silence_ms, self._chunk
        )

    def _chunk(self, text: str, index: int, count: int) -> tuple[np.ndarray, dict, int]:
        start = time.perf_counter()
        audio = self.tts.generate(text).detach().cpu().numpy().astype(np.float32).reshape(-1)
        return audio, {
            "index": index, "attempts": 1,
            "duration": len(audio) / self.sample_rate,
            "t_infer": time.perf_counter() - start, "status": "accepted",
        }, 0
