from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from neutts import NeuTTS

from .. import engine as engine_module
from ..audio import (
    chunk_confidence,
    edge_fade,
    loudness_normalize,
    peak_limit,
    silence,
    to_wav_bytes,
    trim_edge_silence,
)
from ..chunking import split_text
from ..config import Settings
from ..engine import RequestTiming, SynthesisResult
from ..wer import word_error_rate
from .neutts_audio import compute_f0_mean, normalize_f0


class NeuTTSEngine:
    settings: Settings
    tts: NeuTTS | None
    ref_codes: list[int] | None
    ref_text: str | None
    sample_rate: int
    loaded: bool
    model_name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts = None
        self.ref_codes = None
        self.ref_text = None
        self.sample_rate = 24_000
        self.loaded = False
        self.model_name = settings.model

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

        # The perth watermarker leaks substantial allocator memory per call.
        # NeuTTS already treats None as the supported no-watermark path.
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

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        del instructions, speed  # NeuTTS has no equivalent stable controls.

        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        chunks = split_text(text, self.settings.max_chars_per_chunk)
        parts: list[np.ndarray] = []
        chunk_timings: list[dict] = []
        prev_f0_mean: float | None = None
        wer_fallbacks = 0

        try:
            for index, chunk in enumerate(chunks):
                wav, chunk_timing, chunk_fallbacks = self._synthesize_chunk(
                    chunk, index, len(chunks)
                )
                wer_fallbacks += chunk_fallbacks

                f0_start = time.perf_counter()
                if prev_f0_mean is not None:
                    wav = normalize_f0(wav, self.sample_rate, prev_f0_mean)
                prev_f0_mean = compute_f0_mean(wav, self.sample_rate)
                chunk_timing["t_f0"] = time.perf_counter() - f0_start
                chunk_timing["duration"] = float(len(wav) / self.sample_rate)

                if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                    parts.append(
                        silence(
                            self.sample_rate,
                            self.settings.inter_chunk_silence_ms,
                            wav.dtype,
                        )
                    )
                parts.append(wav)
                chunk_timings.append(chunk_timing)

            return SynthesisResult(
                audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
                sample_rate=self.sample_rate,
                chunks=chunks,
                timing=RequestTiming(
                    chunks=chunk_timings,
                    wer_fallbacks=wer_fallbacks,
                ),
            )
        finally:
            # Rerolls temporarily override the persistent GGUF backbone.
            self._apply_generation_settings(self.tts)

    def _synthesize_chunk(
        self,
        chunk_text: str,
        index: int,
        num_chunks: int,
    ) -> tuple[np.ndarray, dict, int]:
        best_audio: np.ndarray | None = None
        best_wer = float("inf")
        best_wer_result: tuple[float, str, bool] | None = None
        best_attempt = -1
        accepted_repeat_penalty = self.settings.repeat_penalty
        wer_fallbacks = 0
        attempt_details: list[dict] = []

        num_attempts = self.settings.max_retries + 1
        for attempt in range(num_attempts):
            rp = (
                self.settings.repeat_penalty
                + attempt * self.settings.repeat_penalty_reroll_step
            )
            self._apply_generation_settings(self.tts, repeat_penalty_override=rp)

            t_infer = t_dsp = t_wer_check = 0.0
            wer: float | None = None
            wer_source: str | None = None
            wer_fallback = False
            accept_early = False

            try:
                infer_start = time.perf_counter()
                try:
                    raw = self.tts.infer(chunk_text, self.ref_codes, self.ref_text)
                finally:
                    t_infer = time.perf_counter() - infer_start

                wav = np.asarray(raw).squeeze()

                dsp_start = time.perf_counter()
                wav = trim_edge_silence(
                    wav,
                    self.sample_rate,
                    leading=index > 0,
                    trailing=index < num_chunks - 1,
                )
                wav = loudness_normalize(wav)
                wav = peak_limit(wav)
                wav = edge_fade(wav, 3.0, self.sample_rate)
                t_dsp = time.perf_counter() - dsp_start

                wer_start = time.perf_counter()
                wer, wer_source, wer_fallback = self._chunk_wer(wav, chunk_text)
                t_wer_check = time.perf_counter() - wer_start
                wer_fallbacks += 1 if wer_fallback else 0

                if wer < best_wer:
                    best_wer = wer
                    best_audio = wav
                    best_wer_result = (wer, wer_source, wer_fallback)
                    best_attempt = attempt
                    accepted_repeat_penalty = rp
                    if wer <= self.settings.wer_threshold:
                        accept_early = True
            except ValueError:
                pass

            attempt_details.append(
                {
                    "attempt": attempt,
                    "t_infer": t_infer,
                    "t_dsp": t_dsp,
                    "t_wer_check": t_wer_check,
                    "wer": wer,
                    "wer_source": wer_source,
                    "wer_fallback": wer_fallback,
                    "repeat_penalty": rp,
                    "accepted": accept_early,
                }
            )

            if accept_early:
                break

        if best_audio is None or best_wer_result is None:
            raise RuntimeError(
                f"All {num_attempts} attempts produced no speech tokens for chunk: {chunk_text!r}"
            )

        if best_attempt >= 0:
            for detail in attempt_details:
                detail["accepted"] = detail["attempt"] == best_attempt

        chunk_wer, chunk_source, chunk_fallback = best_wer_result
        return (
            best_audio,
            {
                "index": index,
                "attempts": len(attempt_details),
                "repeat_penalty": accepted_repeat_penalty,
                "wer": chunk_wer,
                "wer_source": chunk_source,
                "wer_fallback": chunk_fallback,
                "attempts_detail": attempt_details,
            },
            wer_fallbacks,
        )

    def _chunk_wer(self, wav, chunk_text) -> tuple[float, str, bool]:
        if not self.settings.wer_endpoint:
            return 1.0 - chunk_confidence(wav, len(chunk_text)), "confidence", False
        try:
            wav_bytes = to_wav_bytes(wav, self.sample_rate)
            # Keep the historical tinytalk.engine monkeypatch point used by
            # existing tests and tuning helpers while the implementation lives
            # behind this backend boundary.
            transcript = engine_module.transcribe_chunk(
                wav_bytes, chunk_text, self.settings.wer_endpoint
            )
            return word_error_rate(transcript, chunk_text), "whisper", False
        except Exception:
            return 1.0 - chunk_confidence(wav, len(chunk_text)), "confidence", True

    def _validate_reference_files(self) -> None:
        for path_name, path in (
            ("TINYTALK_REF_CODES", self.settings.ref_codes),
            ("TINYTALK_REF_TEXT", self.settings.ref_text),
        ):
            if not Path(path).is_file():
                raise RuntimeError(f"{path_name} does not exist: {path}")
