from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import (
    chunk_confidence,
    compute_f0_mean,
    edge_fade,
    loudness_normalize,
    normalize_f0,
    peak_limit,
    silence,
    to_wav_bytes,
    trim_edge_silence,
)
from .chunking import split_text
from .config import Settings
from .wer import transcribe_chunk, word_error_rate
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

        # The perth watermarker leaks ~44 MB/call: PyTorch's CPU caching allocator
        # never returns the freed STFT/encoder tensors. Nulling it is neutts' own
        # supported no-watermark path (infer guards `watermarker is None`).
        if not self.settings.watermark:
            self.tts.watermarker = None

        self.sample_rate = int(getattr(self.tts, "sample_rate", 24_000))
        self._apply_generation_settings(self.tts)
        self.loaded = True

    def _apply_generation_settings(
        self,
        tts: NeuTTS,
        *,
        repeat_penalty_override: float | None = None,
    ) -> None:
        backbone = tts.backbone
        if not hasattr(backbone, "create_completion"):
            return
        if not hasattr(backbone, "_pristine_create_completion"):
            backbone._pristine_create_completion = backbone.create_completion
        base = backbone._pristine_create_completion
        temperature = self.settings.temperature
        repeat_penalty = (
            repeat_penalty_override
            if repeat_penalty_override is not None
            else self.settings.repeat_penalty
        )

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
            best_audio = None
            best_wer = float("inf")

            num_attempts = self.settings.max_retries + 1
            for attempt in range(num_attempts):
                rp = self.settings.repeat_penalty + attempt * self.settings.repeat_penalty_reroll_step
                self._apply_generation_settings(
                    self.tts, repeat_penalty_override=rp
                )

                try:
                    wav = np.asarray(
                        self.tts.infer(chunk, self.ref_codes, self.ref_text)
                    ).squeeze()

                    wav = trim_edge_silence(
                        wav,
                        self.sample_rate,
                        leading=index > 0,
                        trailing=index < len(chunks) - 1,
                    )

                    wav = loudness_normalize(wav)
                    wav = peak_limit(wav)
                    wav = edge_fade(wav, 3.0, self.sample_rate)

                    wer = self._chunk_wer(wav, chunk)
                    if wer < best_wer:
                        best_wer = wer
                        best_audio = wav
                    if wer <= self.settings.wer_threshold:
                        break
                except ValueError:
                    pass

            if best_audio is None:
                raise RuntimeError(
                    f"All {num_attempts} attempts produced no speech tokens for chunk: {chunk!r}"
                )
            wav = best_audio

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

        # Restore base generation settings after synthesis.
        self._apply_generation_settings(self.tts)

        return SynthesisResult(
            audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
            sample_rate=self.sample_rate,
            chunks=chunks,
        )

    def _chunk_wer(self, wav, chunk_text):
        if not self.settings.wer_endpoint:
            return 1.0 - chunk_confidence(wav, len(chunk_text))
        try:
            wav_bytes = to_wav_bytes(wav, self.sample_rate)
            transcript = transcribe_chunk(wav_bytes, chunk_text, self.settings.wer_endpoint)
            return word_error_rate(transcript, chunk_text)
        except Exception:
            return 1.0 - chunk_confidence(wav, len(chunk_text))

    def _validate_reference_files(self) -> None:
        for path_name, path in (
            ("TINYTALK_REF_CODES", self.settings.ref_codes),
            ("TINYTALK_REF_TEXT", self.settings.ref_text),
        ):
            if not Path(path).is_file():
                raise RuntimeError(f"{path_name} does not exist: {path}")
