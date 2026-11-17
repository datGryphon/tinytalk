from __future__ import annotations

import re
import sys
import time
from typing import TYPE_CHECKING, override

import numpy as np
from huggingface_hub import snapshot_download

from ..chunking import split_tagged_text
from ..config import Settings
from ..engine import ChunkResult, SpeechBackend, SynthesisRequest
from ..timing import ChunkTiming

if TYPE_CHECKING:
    from silencio2 import Silencio2TTS

_REPO_ID = "BlueRoseLabs/Silencio-2"
_REVISION = "20930f5f6a6bb564f2a8207152fd5c5d8eecff03"

_TAGS_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")
_EMOTION_RE = re.compile(r"<([^\s<>]+)>")


class Silencio2Backend(SpeechBackend[SynthesisRequest]):
    tts: Silencio2TTS | None
    _emotions: tuple[str, ...]
    _sounds: tuple[str, ...]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, _REPO_ID)
        self.tts = None
        self._emotions = ()
        self._sounds = ()

    @override
    def load(self) -> None:
        if self.loaded:
            return

        snapshot = snapshot_download(repo_id=_REPO_ID, revision=_REVISION)
        # Upstream ships its package without Python distribution metadata.
        sys.path.insert(0, snapshot)
        try:
            import silencio2
        finally:
            sys.path.remove(snapshot)

        self._emotions = tuple(silencio2.EMOTIONS)
        self._sounds = tuple(silencio2.SOUNDS)

        self.tts = silencio2.Silencio2TTS.from_pretrained(snapshot, device="cpu")
        self.sample_rate = int(self.tts.sample_rate)

        if self.voices is None:
            self.tts.voice(self.settings.silencio2_voice)
        else:
            for preset in self.voices.voices.values():
                assert preset.voice_id is not None
                self.tts.voice(preset.voice_id)
        self.loaded = True

    @override
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
    ) -> SynthesisRequest:
        if instructions and instructions.strip():
            raise ValueError(
                "silencio2 controls delivery with inline text tags, not instructions"
            )
        if speed is not None and speed != 1.0:
            raise ValueError("silencio2 supports only speed=1.0")

        return SynthesisRequest(instructions=None, speed=None)

    @override
    def prepare_chunks(self, text: str, max_chars_per_chunk: int | None = None) -> list[str]:
        self._validate_tags(text)
        chunks = split_tagged_text(text, max_chars_per_chunk or self.settings.max_chars_per_chunk)
        return self._carry_emotions(chunks)

    def _validate_tags(self, text: str) -> None:
        for full in _TAGS_RE.findall(text):
            content = full[1:-1]
            if full[0] == "[":
                if content not in self._sounds:
                    raise ValueError(f"unknown sound tag [{content}]")
            elif content not in self._emotions:
                raise ValueError(f"unknown emotion tag <{content}>")

    def _carry_emotions(self, chunks: list[str]) -> list[str]:
        active: str | None = None
        out: list[str] = []
        for chunk in chunks:
            prefix = f"<{active}> " if active and not _EMOTION_RE.match(chunk) else ""
            out.append(prefix + chunk)
            emotions = _EMOTION_RE.findall(chunk)
            if emotions:
                active = emotions[-1]
        return out

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
        wav = self.tts.generate(
            chunk_text,
            voice=(request.voice.voice_id if request.voice is not None else self.settings.silencio2_voice),
            normalize=False
        )
        t_infer = time.perf_counter() - start

        audio = wav.detach().cpu().numpy().astype(np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        timing = ChunkTiming(
            index=index,
            attempts=1,
            duration=len(audio) / self.sample_rate,
            t_infer=t_infer,
            status="accepted",
        )
        return audio, timing, 0
