from __future__ import annotations

import json
import time

import numpy as np
from tinytauk import TinyTAuK

from .. import engine as engine_module
from ..audio import edge_fade, peak_limit, silence, trim_edge_silence
from ..chunking import split_text
from ..config import Settings
from ..engine import RequestTiming, SynthesisResult
from ..quality import QualityResult, evaluate_audio, quality_is_acceptable, quality_rank

_DEFAULT_DESCRIPTION = "A clear, natural speaking voice"
_PROMPT_SCAFFOLD = "Based on the following description, generate speech content."
_WARMUP_TEXT = (
    "TinyTalk is warming the speech runtime before serving requests so the first user synthesis "
    "runs on the compiled path."
)
_WARMUP_SECONDS = 9.0
_DURATION_SHRINK_INSERTIONS = 0.92
_DURATION_GROW_DELETIONS = 1.10
_DURATION_MIN_FACTOR = 0.75
_DURATION_MAX_FACTOR = 1.25


class TinyTAuKEngine:
    settings: Settings
    tts: TinyTAuK | None
    sample_rate: int
    loaded: bool
    model_name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts = None
        self.sample_rate = 24_000
        self.loaded = False
        self.model_name = settings.tinytauk_model

    def load(self) -> None:
        if self.settings.tinytauk_chars_per_second <= 0:
            raise ValueError("TINYTALK_TINYTAUK_CHARS_PER_SECOND must be positive")

        self.tts = TinyTAuK.from_pretrained(
            model_id=self.settings.tinytauk_model,
            qwen_model_id=self.settings.tinytauk_qwen_model,
        )

        # TinyTAuK's VAE compiles lazily. Exercise the same full-generation path
        # used for real requests before the service reports healthy.
        warmup = self.tts.generate(
            self._instruction(_WARMUP_TEXT, None),
            gen_seconds=_WARMUP_SECONDS,
        )
        self.sample_rate = int(warmup.sample_rate)
        self.loaded = True

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        speed_value = 1.0 if speed is None else speed
        if speed_value <= 0:
            raise ValueError("speed must be positive")

        chunks = split_text(text, self.settings.max_chars_per_chunk)
        parts: list[np.ndarray] = []
        chunk_timings: list[dict] = []
        wer_fallbacks = 0

        for index, chunk in enumerate(chunks):
            wav, chunk_timing, chunk_fallbacks = self._synthesize_chunk(
                chunk,
                index=index,
                num_chunks=len(chunks),
                instructions=instructions,
                speed=speed_value,
            )
            wer_fallbacks += chunk_fallbacks

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

    def _synthesize_chunk(
        self,
        chunk_text: str,
        *,
        index: int,
        num_chunks: int,
        instructions: str | None,
        speed: float,
    ) -> tuple[np.ndarray, dict, int]:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        description = self._description(instructions)
        base_duration = self._duration_seconds(chunk_text, speed)
        duration = base_duration
        base_seed = self.tts.config.runtime.seed
        num_attempts = self.settings.max_retries + 1

        best_audio: np.ndarray | None = None
        best_quality: QualityResult | None = None
        best_rank: tuple[bool, bool, float, float] | None = None
        best_attempt = -1
        best_seed: int | None = None
        best_duration = duration
        wer_fallbacks = 0
        attempt_details: list[dict] = []

        for attempt in range(num_attempts):
            seed = self._attempt_seed(base_seed, index, attempt, num_attempts)
            t_infer = 0.0
            t_dsp = 0.0
            t_wer_check = 0.0
            quality: QualityResult | None = None
            accept_early = False

            try:
                infer_start = time.perf_counter()
                try:
                    result = self.tts.generate(
                        self._instruction(chunk_text, instructions),
                        gen_seconds=duration,
                        seed=seed,
                    )
                finally:
                    t_infer = time.perf_counter() - infer_start

                if int(result.sample_rate) != self.sample_rate:
                    raise RuntimeError(
                        f"TinyTAuK sample rate changed from {self.sample_rate} to {result.sample_rate}"
                    )

                dsp_start = time.perf_counter()
                wav = (
                    result.audio.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                    .squeeze()
                )
                wav = trim_edge_silence(
                    wav,
                    self.sample_rate,
                    leading=index > 0,
                    trailing=index < num_chunks - 1,
                )
                # Preserve AuK's requested loudness/prosody. Only guard clipping and
                # soften splice edges; NeuTTS-specific RMS/F0 normalization is not used.
                wav = peak_limit(wav)
                wav = edge_fade(wav, 3.0, self.sample_rate)
                t_dsp = time.perf_counter() - dsp_start

                quality_start = time.perf_counter()
                quality = evaluate_audio(
                    wav,
                    self.sample_rate,
                    target_text=chunk_text,
                    endpoint=self.settings.wer_endpoint,
                    prompt_text=description,
                    scaffold_text=_PROMPT_SCAFFOLD,
                    transcribe=engine_module.transcribe_chunk,
                )
                t_wer_check = time.perf_counter() - quality_start
                wer_fallbacks += int(quality.fallback)

                accept_early = quality_is_acceptable(
                    quality,
                    self.settings.wer_threshold,
                )
                rank = quality_rank(quality)
                if accept_early or best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_audio = wav
                    best_quality = quality
                    best_attempt = attempt
                    best_seed = seed
                    best_duration = duration
            except ValueError:
                pass

            detail = {
                "attempt": attempt,
                "t_infer": t_infer,
                "t_dsp": t_dsp,
                "t_wer_check": t_wer_check,
                "repeat_penalty": None,
                "seed": seed,
                "gen_seconds": duration,
                "status": None,
            }
            detail.update(self._quality_timing_fields(quality))
            attempt_details.append(detail)

            if accept_early:
                break
            if quality is not None:
                duration = self._adjust_duration(base_duration, duration, quality)

        if best_audio is None or best_quality is None:
            raise RuntimeError(
                f"All {num_attempts} attempts produced no usable audio for chunk: {chunk_text!r}"
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
            "duration": float(len(best_audio) / self.sample_rate),
            "repeat_penalty": None,
            "seed": best_seed,
            "gen_seconds": best_duration,
            "t_f0": 0.0,
            "attempts_detail": attempt_details,
        }
        chunk_timing.update(best_quality.timing_fields())
        return best_audio, chunk_timing, wer_fallbacks

    def _duration_seconds(self, text: str, speed: float) -> float:
        seconds = len(text) / self.settings.tinytauk_chars_per_second / speed
        return max(1.0, seconds)

    @staticmethod
    def _attempt_seed(base_seed: int, chunk_index: int, attempt: int, attempts_per_chunk: int) -> int:
        return base_seed + chunk_index * attempts_per_chunk + attempt

    @staticmethod
    def _adjust_duration(
        base_duration: float,
        current_duration: float,
        quality: QualityResult,
    ) -> float:
        # Prompt leakage can replace requested words at the beginning/middle of
        # an utterance, not merely consume spare time at the end. Shortening that
        # case risks truncating even more requested speech; reroll it by seed and
        # preserve the caller-derived duration. Duration correction is reserved
        # for non-leak edit shapes that actually indicate over/under generation.
        if quality.prompt_leak:
            adjusted = current_duration
        elif (
            quality.insertions is not None
            and quality.deletions is not None
            and quality.insertions > quality.deletions
        ):
            adjusted = current_duration * _DURATION_SHRINK_INSERTIONS
        elif (
            quality.insertions is not None
            and quality.deletions is not None
            and quality.deletions > quality.insertions
        ):
            adjusted = current_duration * _DURATION_GROW_DELETIONS
        else:
            adjusted = current_duration

        lower = max(1.0, base_duration * _DURATION_MIN_FACTOR)
        upper = max(lower, base_duration * _DURATION_MAX_FACTOR)
        return min(max(adjusted, lower), upper)

    @staticmethod
    def _description(instructions: str | None) -> str:
        if instructions and instructions.strip():
            return instructions.strip()
        return _DEFAULT_DESCRIPTION

    @classmethod
    def _instruction(cls, text: str, instructions: str | None) -> str:
        quoted_description = json.dumps(cls._description(instructions), ensure_ascii=False)
        quoted_text = json.dumps(text, ensure_ascii=False)
        return (
            f"Based on the following description: {quoted_description}, "
            f"generate speech content {quoted_text}."
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
