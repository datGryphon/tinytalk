from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

from ..chunking import split_tagged_text
from ..config import Settings
from ..engine import SynthesisResult, synthesize_chunks

_REPO_ID = "BlueRoseLabs/Silencio-2"
_REVISION = "20930f5f6a6bb564f2a8207152fd5c5d8eecff03"

_TAGS_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")
_EMOTION_RE = re.compile(r"<([^\s<>]+)>")


def _assert_snapshot_owned(silencio2_module: Any, snapshot_path: Path) -> None:
    filename = getattr(silencio2_module, "__file__", None)
    if not filename:
        raise RuntimeError("silencio2 has no verifiable snapshot origin")
    module_file = Path(filename)
    try:
        module_file.absolute().parent.relative_to(snapshot_path)
    except ValueError:
        raise RuntimeError(
            "silencio2 loaded from outside the pinned snapshot; a conflicting "
            "installation shadows it"
        ) from None


class Silencio2Engine:
    settings: Settings
    tts: Any | None
    sample_rate: int
    loaded: bool
    model_name: str
    _emotions: tuple[str, ...]
    _sounds: tuple[str, ...]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts = None
        self.sample_rate = 24_000
        self.loaded = False
        self.model_name = "BlueRoseLabs/Silencio-2"
        self._emotions = ()
        self._sounds = ()

    def load(self) -> None:
        if self.loaded:
            return

        snapshot = snapshot_download(repo_id=_REPO_ID, revision=_REVISION)
        snapshot_path = Path(snapshot).resolve()
        # Upstream ships its package without Python distribution metadata.
        sys.path.insert(0, str(snapshot_path))
        try:
            import silencio2
        finally:
            sys.path.remove(str(snapshot_path))

        _assert_snapshot_owned(silencio2, snapshot_path)
        self._emotions = tuple(silencio2.EMOTIONS)
        self._sounds = tuple(silencio2.SOUNDS)

        self.tts = silencio2.Silencio2TTS.from_pretrained(str(snapshot_path), device="cpu")
        self.sample_rate = int(self.tts.sample_rate)

        self.tts.voice(self.settings.silencio2_voice)
        self.loaded = True

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        if not self.loaded:
            raise RuntimeError("engine not loaded. Call load() first")
        if not text or not text.strip():
            raise ValueError("cannot synthesize empty text")
        if instructions and instructions.strip():
            raise ValueError(
                "silencio2 controls delivery with inline text tags, not instructions"
            )
        if speed is not None and speed != 1.0:
            raise ValueError("silencio2 supports only speed=1.0")

        chunks = self._prepare_chunks(text)
        return synthesize_chunks(
            chunks,
            self.sample_rate,
            self.settings.inter_chunk_silence_ms,
            self._synthesize_chunk,
        )

    def _prepare_chunks(self, text: str) -> list[str]:
        self._validate_tags(text)
        chunks = split_tagged_text(text, self.settings.max_chars_per_chunk)
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

    def _synthesize_chunk(
        self,
        chunk_text: str,
        index: int,
        num_chunks: int,
    ) -> tuple[np.ndarray, dict, int]:
        if not self.loaded:
            raise RuntimeError("engine not loaded. Call load() first")

        start = time.perf_counter()
        wav = self.tts.generate(
            chunk_text, voice=self.settings.silencio2_voice, normalize=False
        )
        t_infer = time.perf_counter() - start

        audio = wav.detach().cpu().numpy().astype(np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        timing = {
            "index": index,
            "text": chunk_text,
            "attempts": 1,
            "duration": float(len(audio) / self.sample_rate),
            "t_infer": t_infer,
            "status": "accepted",
        }
        return audio, timing, 0
