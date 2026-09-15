from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from neutts import NeuTTS

from .. import engine as engine_module
from ..audio import (
    edge_fade,
    loudness_normalize,
    peak_limit,
    silence,
    trim_edge_silence,
)
from ..chunking import split_text
from ..config import Settings
from ..engine import RequestTiming, SynthesisResult
from ..quality import QualityResult, evaluate_audio, quality_is_acceptable, quality_rank
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
        best_quality: QualityResult | None = None
        best_rank: tuple[bool, bool, float, float] | None = None
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

            t_infer = 0.0
            t_dsp = 0.0
            t_wer_check = 0.0
            quality: QualityResult | None = None
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

                quality_start = time.perf_counter()
                quality = self._chunk_quality(wav, chunk_text)
                t_wer_check = time.perf_counter() - quality_start
                wer_fallbacks += int(quality.fallback)

                accept_early = quality_is_acceptable(
                    quality,
                    self.settings.wer_threshold,
                )
                rank = quality_rank(quality)
                if accept_early or best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_quality = quality
                    best_audio = wav
                    best_attempt = attempt
                    accepted_repeat_penalty = rp
            except ValueError:
                pass

            detail = {
                "attempt": attempt,
                "t_infer": t_infer,
                "t_dsp": t_dsp,
                "t_wer_check": t_wer_check,
                "repeat_penalty": rp,
                "status": None,
            }
            detail.update(self._quality_timing_fields(quality))
            attempt_details.append(detail)

            if accept_early:
                break

        if best_audio is None or best_quality is None:
            raise RuntimeError(
                f"All {num_attempts} attempts produced no speech tokens for chunk: {chunk_text!r}"
            )

        selected_status = (
            "accepted"
            if quality_is_acceptable(best_quality, self.settings.wer_threshold)
            else "fallback"
        )
        for detail in attempt_details:
            if detail["attempt"] == best_attempt:
                detail["status"] = selected_status

        chunk_timing = {
            "index": index,
            "attempts": len(attempt_details),
            "status": selected_status,
            "repeat_penalty": accepted_repeat_penalty,
            "attempts_detail": attempt_details,
        }
        chunk_timing.update(best_quality.timing_fields())
        return best_audio, chunk_timing, wer_fallbacks

    def _chunk_quality(self, wav: np.ndarray, chunk_text: str) -> QualityResult:
        # Route through the historical _chunk_wer seam so existing tests and
        # tuning helpers can still replace it while quality scoring is shared.
        result = self._chunk_wer(wav, chunk_text)
        if isinstance(result, QualityResult):
            return result

        wer, source, fallback = result
        return QualityResult(
            wer=float(wer),
            cer=None,
            substitutions=None,
            deletions=None,
            insertions=None,
            prompt_leak=False,
            source=str(source),
            fallback=bool(fallback),
        )

    def _chunk_wer(self, wav: np.ndarray, chunk_text: str):
        return evaluate_audio(
            wav,
            self.sample_rate,
            target_text=chunk_text,
            endpoint=self.settings.wer_endpoint,
            transcribe=engine_module.transcribe_chunk,
        )

    @staticmethod
    def _quality_timing_fields(quality: QualityResult | None) -> dict[str, object]:
        if quality is None:
            return {
                "wer": None,
                "cer": None,
                "substitutions": None,
                "deletions": None,
                "insertions": None,
                "prompt_leak": False,
                "wer_source": None,
                "wer_fallback": False,
            }
        return quality.timing_fields()

    def _validate_reference_files(self) -> None:
        for path_name, path in (
            ("TINYTALK_REF_CODES", self.settings.ref_codes),
            ("TINYTALK_REF_TEXT", self.settings.ref_text),
        ):
            if not Path(path).is_file():
                raise RuntimeError(f"{path_name} does not exist: {path}")
