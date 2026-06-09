from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import silence, trim_edge_silence
from .chunking import split_text
from .config import Settings


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    chunks: list[str]


class TinyTalkEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tts: Any | None = None
        self.ref_codes: list[int] | None = None
        self.ref_text: str | None = None
        self.sample_rate = 24_000
        self.loaded = False

    def load(self) -> None:
        self._validate_reference_files()

        import torch

        self.ref_codes = (
            torch.load(self.settings.ref_codes, map_location="cpu")
            .detach()
            .cpu()
            .flatten()
            .to(torch.long)
            .tolist()
        )
        self.ref_text = self.settings.ref_text.read_text(encoding="utf-8").strip()

        if not self.ref_text:
            raise RuntimeError(f"reference text is empty: {self.settings.ref_text}")

        from neutts import NeuTTS

        self.tts = NeuTTS(
            backbone_repo=self.settings.model,
            backbone_device=self.settings.backbone_device,
            codec_repo=self.settings.codec,
            codec_device="cpu",
        )
        self.sample_rate = int(getattr(self.tts, "sample_rate", 24_000))
        self.loaded = True

    def synthesize(self, text: str) -> SynthesisResult:
        chunks = split_text(text, self.settings.max_chars_per_chunk)
        parts: list[np.ndarray] = []
        trim_chunks = len(chunks) > 1

        for index, chunk in enumerate(chunks):
            wav = np.asarray(
                self.tts.infer(chunk, self.ref_codes, self.ref_text)
            ).squeeze()
            if trim_chunks:
                wav = trim_edge_silence(
                    wav,
                    self.sample_rate,
                    leading=index > 0,
                    trailing=index < len(chunks) - 1,
                )

            if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                parts.append(
                    silence(
                        self.sample_rate,
                        self.settings.inter_chunk_silence_ms,
                        wav.dtype,
                    )
                )
            parts.append(wav)

        return SynthesisResult(
            audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
            sample_rate=self.sample_rate,
            chunks=chunks,
        )

    def _validate_reference_files(self) -> None:
        for path_name, path in (
            ("TINYTALK_REF_CODES", self.settings.ref_codes),
            ("TINYTALK_REF_TEXT", self.settings.ref_text),
        ):
            if not Path(path).is_file():
                raise RuntimeError(f"{path_name} does not exist: {path}")
