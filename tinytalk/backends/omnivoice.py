from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omnivoice.models.omnivoice import OmniVoice

from .. import engine as engine_module
from ..audio import edge_fade, peak_limit, silence, trim_edge_silence
from ..chunking import split_text
from ..config import Settings
from ..engine import RequestTiming, SynthesisResult
from ..quality import QualityResult, evaluate_audio, quality_is_acceptable, quality_rank

_RETRY_CLASS_TEMPERATURE_STEP = 0.2
_MAX_RETRY_CLASS_TEMPERATURE = 1.0


class OmniVoiceEngine:
    settings: Settings
    tts: OmniVoice | None
    voice_clone_prompt: Any | None
    sample_rate: int
    loaded: bool
    model_name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts = None
        self.voice_clone_prompt = None
        self.sample_rate = 24_000
        self.loaded = False
        self.model_name = settings.omnivoice_model

    def load(self) -> None:
        self._validate_reference_config()
        device = self.settings.omnivoice_device.strip() or "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.tts = OmniVoice.from_pretrained(
            self.settings.omnivoice_model,
            device_map=device,
            dtype=dtype,
        )
        self.sample_rate = int(self.tts.sampling_rate)

        if self.settings.omnivoice_ref_audio is not None:
            ref_text_path = self._required_ref_text_path()
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
            if not ref_text:
                raise ValueError("TINYTALK_OMNIVOICE_REF_TEXT must contain a transcript")
            self.voice_clone_prompt = self.tts.create_voice_clone_prompt(
                ref_audio=str(self.settings.omnivoice_ref_audio),
                ref_text=ref_text,
            )

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
        if speed is not None and speed <= 0:
            raise ValueError("speed must be positive")

        instructions = instructions.strip() if instructions else None
        language = self.settings.omnivoice_language.strip() or None
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
                speed=speed,
                language=language,
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
            timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=wer_fallbacks),
        )

    def _synthesize_chunk(
        self,
        chunk_text: str,
        *,
        index: int,
        num_chunks: int,
        instructions: str | None,
        speed: float | None,
        language: str | None,
    ) -> tuple[np.ndarray, dict, int]:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        mode = self._mode(instructions)
        num_attempts = self.settings.max_retries + 1
        best_audio: np.ndarray | None = None
        best_quality: QualityResult | None = None
        best_attempt = -1
        best_temperature = 0.0
        wer_fallbacks = 0
        attempt_details: list[dict] = []

        for attempt in range(num_attempts):
            temperature = min(
                _MAX_RETRY_CLASS_TEMPERATURE,
                attempt * _RETRY_CLASS_TEMPERATURE_STEP,
            )
            wav, quality, detail = self._run_attempt(
                chunk_text,
                attempt=attempt,
                index=index,
                num_chunks=num_chunks,
                mode=mode,
                instructions=instructions,
                speed=speed,
                language=language,
                temperature=temperature,
            )
            wer_fallbacks += int(quality.fallback)

            accepted = quality_is_acceptable(quality, self.settings.wer_threshold)
            if (
                accepted
                or best_quality is None
                or quality_rank(quality) < quality_rank(best_quality)
            ):
                best_audio = wav
                best_quality = quality
                best_attempt = attempt
                best_temperature = temperature

            attempt_details.append(detail)
            if accepted:
                break

        if best_audio is None or best_quality is None:
            raise RuntimeError(f"All {num_attempts} attempts produced no usable OmniVoice audio")

        chunk_timing = self._chunk_timing(
            index=index,
            mode=mode,
            audio=best_audio,
            quality=best_quality,
            temperature=best_temperature,
            selected_attempt=best_attempt,
            details=attempt_details,
        )
        return best_audio, chunk_timing, wer_fallbacks

    def _run_attempt(
        self,
        chunk_text: str,
        *,
        attempt: int,
        index: int,
        num_chunks: int,
        mode: str,
        instructions: str | None,
        speed: float | None,
        language: str | None,
        temperature: float,
    ) -> tuple[np.ndarray, QualityResult, dict]:
        raw_audio, t_infer = self._generate_audio(
            chunk_text,
            mode=mode,
            instructions=instructions,
            language=language,
            speed=speed,
            temperature=temperature,
        )
        wav, t_dsp = self._postprocess_audio(
            raw_audio,
            index=index,
            num_chunks=num_chunks,
        )
        quality, t_wer_check = self._evaluate_quality(
            wav,
            chunk_text,
            mode=mode,
            instructions=instructions,
        )

        detail = {
            "attempt": attempt,
            "t_infer": t_infer,
            "t_dsp": t_dsp,
            "t_wer_check": t_wer_check,
            "repeat_penalty": None,
            "class_temperature": temperature,
            "mode": mode,
            "status": None,
        }
        detail.update(quality.timing_fields())
        return wav, quality, detail

    def _generate_audio(
        self,
        text: str,
        *,
        mode: str,
        instructions: str | None,
        language: str | None,
        speed: float | None,
        temperature: float,
    ) -> tuple[np.ndarray, float]:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        start = time.perf_counter()
        audios = self.tts.generate(
            text=text,
            language=language,
            voice_clone_prompt=self.voice_clone_prompt if mode == "clone" else None,
            instruct=instructions if mode == "design" else None,
            speed=speed,
            class_temperature=temperature,
            postprocess_output=True,
        )
        elapsed = time.perf_counter() - start
        if not audios:
            raise ValueError("OmniVoice returned no audio")
        return audios[0], elapsed

    def _postprocess_audio(
        self,
        audio: np.ndarray,
        *,
        index: int,
        num_chunks: int,
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        wav = np.asarray(audio, dtype=np.float32).squeeze()
        if wav.ndim != 1 or wav.size == 0:
            raise ValueError("OmniVoice returned invalid audio shape")

        wav = trim_edge_silence(
            wav,
            self.sample_rate,
            leading=index > 0,
            trailing=index < num_chunks - 1,
        )
        wav = peak_limit(wav)
        wav = edge_fade(wav, 3.0, self.sample_rate)
        return wav, time.perf_counter() - start

    def _evaluate_quality(
        self,
        wav: np.ndarray,
        chunk_text: str,
        *,
        mode: str,
        instructions: str | None,
    ) -> tuple[QualityResult, float]:
        start = time.perf_counter()
        quality = evaluate_audio(
            wav,
            self.sample_rate,
            target_text=chunk_text,
            endpoint=self.settings.wer_endpoint,
            prompt_text=instructions if mode == "design" else "",
            scaffold_text="",
            transcribe=engine_module.transcribe_chunk,
        )
        return quality, time.perf_counter() - start

    def _chunk_timing(
        self,
        *,
        index: int,
        mode: str,
        audio: np.ndarray,
        quality: QualityResult,
        temperature: float,
        selected_attempt: int,
        details: list[dict],
    ) -> dict:
        status = (
            "accepted"
            if quality_is_acceptable(quality, self.settings.wer_threshold)
            else "fallback"
        )
        details[selected_attempt]["status"] = status

        timing = {
            "index": index,
            "attempts": len(details),
            "status": status,
            "duration": float(len(audio) / self.sample_rate),
            "repeat_penalty": None,
            "class_temperature": temperature,
            "mode": mode,
            "t_f0": 0.0,
            "attempts_detail": details,
        }
        timing.update(quality.timing_fields())
        return timing

    def _mode(self, instructions: str | None) -> str:
        if instructions:
            return "design"
        if self.voice_clone_prompt is not None:
            return "clone"
        return "auto"

    def _validate_reference_config(self) -> None:
        has_audio = self.settings.omnivoice_ref_audio is not None
        has_text = self.settings.omnivoice_ref_text is not None
        if has_audio != has_text:
            raise ValueError(
                "TINYTALK_OMNIVOICE_REF_AUDIO and TINYTALK_OMNIVOICE_REF_TEXT "
                "must be configured together"
            )
        for path in (self.settings.omnivoice_ref_audio, self.settings.omnivoice_ref_text):
            if path is not None and not path.is_file():
                raise ValueError(f"OmniVoice reference file does not exist: {path}")

    def _required_ref_text_path(self) -> Path:
        if self.settings.omnivoice_ref_text is None:
            raise RuntimeError("OmniVoice reference text is not configured")
        return self.settings.omnivoice_ref_text

