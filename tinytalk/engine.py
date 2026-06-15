from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import compute_f0_mean, normalize_f0, silence, trim_edge_silence
from .chunking import split_text
from .config import Settings
from neutts import NeuTTS


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    chunks: list[str]


class TinyTalkEngine:

    settings: Settings
    tts: NeuTTS | None
    ref_codes: list[int] | None
    ref_text: str | None
    sample_rate: int
    loaded: bool

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts = None
        self.ref_codes = None
        self.ref_text = None
        self.sample_rate = 24_000
        self.loaded = False

    def load(self) -> None:
        self._validate_reference_files()

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

        self.tts = NeuTTS(
            backbone_repo=self.settings.model,
            backbone_device=self.settings.backbone_device,
            codec_repo=self.settings.codec,
            codec_device="cpu",
        )
        self.sample_rate = int(getattr(self.tts, "sample_rate", 24_000))
        self._apply_generation_settings(self.tts)
        self.loaded = True

    def _apply_generation_settings(self, tts: NeuTTS) -> None:
        # NeuTTS hardcodes temperature/repeat_penalty and exposes no override, so
        # inject ours into the GGUF backbone's create_completion call.
        backbone = tts.backbone
        if not hasattr(backbone, "create_completion"):
            return
        base = backbone.create_completion
        temperature = self.settings.temperature
        repeat_penalty = self.settings.repeat_penalty

        def create_completion(*args, **kwargs):
            kwargs["temperature"] = temperature
            kwargs["repeat_penalty"] = repeat_penalty
            return base(*args, **kwargs)

        backbone.create_completion = create_completion

    def synthesize(self, text: str) -> SynthesisResult:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        chunks = split_text(text, self.settings.max_chars_per_chunk)
        parts: list[np.ndarray] = []

        prev_f0_mean: float | None = None

        for index, chunk in enumerate(chunks):
            wav = np.asarray(
                self.tts.infer(chunk, self.ref_codes, self.ref_text)
            ).squeeze()
            wav = trim_edge_silence(
                wav,
                self.sample_rate,
                leading=index > 0,
                trailing=index < len(chunks) - 1,
            )

            # Smooth pitch across boundaries: nudge each chunk toward the previous
            # chunk's pitch, letting it drift naturally over long text.
            if prev_f0_mean is not None:
                wav = normalize_f0(wav, self.sample_rate, prev_f0_mean)

            if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                parts.append(
                    silence(
                        self.sample_rate,
                        self.settings.inter_chunk_silence_ms,
                        wav.dtype,
                    )
                )
            parts.append(wav)

            prev_f0_mean = compute_f0_mean(wav, self.sample_rate)

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
