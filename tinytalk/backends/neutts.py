from __future__ import annotations

import time
from pathlib import Path
from typing import override

import numpy as np
import torch
from neutts import NeuTTS

from .. import engine as engine_module
from ..audio import (
    AudioSamples,
    edge_fade,
    loudness_normalize,
    peak_limit,
    trim_edge_silence,
)
from ..config import Settings
from ..engine import ChunkResult, SpeechBackend, SynthesisRequest
from ..quality import QualityResult, evaluate_audio, quality_is_acceptable, quality_rank
from ..timing import AttemptStatus, AttemptTiming, ChunkTiming, QualityTiming
from .neutts_audio import compute_f0_mean, normalize_f0


class NeuTTSRequest(SynthesisRequest):
    previous_f0_mean: float | None = None


class NeuTTSBackend(SpeechBackend[NeuTTSRequest]):
    tts: NeuTTS | None
    ref_codes: list[int] | None
    ref_text: str | None
    voice_references: dict[Path, tuple[list[int], str]]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.model)
        self.tts = None
        self.ref_codes = None
        self.ref_text = None
        self.voice_references = {}

    @override
    def load(self) -> None:
        if self.voices is None:
            self._validate_reference_files()
            self.ref_codes, self.ref_text = self._read_reference(
                self.settings.ref_codes, self.settings.ref_text
            )
        else:
            for preset in self.voices.voices.values():
                assert preset.ref_codes is not None and preset.ref_text is not None
                self.voice_references[preset.ref_codes] = self._read_reference(
                    preset.ref_codes, preset.ref_text
                )

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

    @override
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
    ) -> NeuTTSRequest:
        del instructions, speed  # NeuTTS has no equivalent stable controls.

        return NeuTTSRequest(instructions=None, speed=None)

    @override
    def synthesize_chunk(
        self,
        request: NeuTTSRequest,
        chunk_text: str,
        index: int,
        num_chunks: int,
    ) -> ChunkResult:
        if request.voice is not None and request.voice.ref_codes is not None:
            reference = self.voice_references[request.voice.ref_codes]
            wav, timing, fallbacks = self._synthesize_chunk(
                chunk_text, index, num_chunks, reference=reference
            )
        else:
            wav, timing, fallbacks = self._synthesize_chunk(chunk_text, index, num_chunks)
        f0_start = time.perf_counter()
        if request.previous_f0_mean is not None:
            wav = normalize_f0(wav, self.sample_rate, request.previous_f0_mean)
        request.previous_f0_mean = compute_f0_mean(wav, self.sample_rate)
        timing.t_f0 = time.perf_counter() - f0_start
        timing.duration = len(wav) / self.sample_rate
        return wav, timing, fallbacks

    @override
    def finalize_request(self, request: NeuTTSRequest) -> None:
        # Rerolls temporarily override the persistent GGUF backbone.
        if self.tts is not None:
            self._apply_generation_settings(self.tts)

    def _synthesize_chunk(
        self,
        chunk_text: str,
        index: int,
        num_chunks: int,
        *,
        reference: tuple[list[int] | None, str | None] | None = None,
    ) -> ChunkResult:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")
        best_audio: AudioSamples | None = None
        best_quality: QualityResult | None = None
        best_rank: tuple[bool, bool, float, float] | None = None
        best_attempt = -1
        accepted_repeat_penalty = self.settings.repeat_penalty
        wer_fallbacks = 0
        attempt_details: list[AttemptTiming] = []

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
                    raw = self.tts.infer(chunk_text, *(reference or (self.ref_codes, self.ref_text)))
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
                quality = self._chunk_wer(wav, chunk_text)
                t_wer_check = time.perf_counter() - quality_start
                wer_fallbacks += int(quality.wer_fallback)

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

            attempt_details.append(AttemptTiming(
                attempt=attempt,
                t_infer=t_infer,
                t_dsp=t_dsp,
                t_wer_check=t_wer_check,
                repeat_penalty=rp,
                status=None,
                **(quality or QualityTiming()).model_dump(),
            ))

            if accept_early:
                break

        if best_audio is None or best_quality is None:
            raise RuntimeError(
                f"All {num_attempts} attempts produced no speech tokens for chunk: {chunk_text!r}"
            )

        selected_status: AttemptStatus = (
            "accepted"
            if quality_is_acceptable(best_quality, self.settings.wer_threshold)
            else "fallback"
        )
        attempt_details[best_attempt].status = selected_status

        chunk_timing = ChunkTiming(
            index=index,
            attempts=len(attempt_details),
            status=selected_status,
            repeat_penalty=accepted_repeat_penalty,
            duration=len(best_audio) / self.sample_rate,
            attempts_detail=attempt_details,
            **best_quality.model_dump(),
        )
        return best_audio, chunk_timing, wer_fallbacks

    def _chunk_wer(self, wav: AudioSamples, chunk_text: str) -> QualityResult:
        return evaluate_audio(
            wav,
            self.sample_rate,
            target_text=chunk_text,
            endpoint=self.settings.wer_endpoint,
            transcribe=engine_module.transcribe_chunk,
        )

    def _validate_reference_files(self) -> None:
        for path_name, path in (
            ("TINYTALK_REF_CODES", self.settings.ref_codes),
            ("TINYTALK_REF_TEXT", self.settings.ref_text),
        ):
            if not Path(path).is_file():
                raise RuntimeError(f"{path_name} does not exist: {path}")

    @staticmethod
    def _read_reference(codes_path: Path, text_path: Path) -> tuple[list[int], str]:
        for path in (codes_path, text_path):
            if not path.is_file():
                raise ValueError(f"NeuTTS reference file does not exist: {path}")
        codes = (
            torch.load(codes_path, map_location="cpu")
            .detach().cpu().flatten().to(torch.long).tolist()
        )
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"reference text is empty: {text_path}")
        return codes, text
