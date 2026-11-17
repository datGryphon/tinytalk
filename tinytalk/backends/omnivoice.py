from __future__ import annotations

import time
from typing import override

import numpy as np
import torch
from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt

from .. import engine as engine_module
from ..audio import AudioArray, AudioSamples, edge_fade, peak_limit, trim_edge_silence
from ..config import OmniVoicePreset, Settings, VoiceConfig, load_voice_config
from ..engine import ChunkResult, SpeechBackend, SynthesisRequest
from ..quality import QualityResult, evaluate_audio, quality_is_acceptable, quality_rank
from ..timing import AttemptStatus, AttemptTiming, ChunkTiming, VoiceMode

_RETRY_CLASS_TEMPERATURE_STEP = 0.2
_MAX_RETRY_CLASS_TEMPERATURE = 1.0


class OmniVoiceBackend(SpeechBackend[SynthesisRequest]):
    tts: OmniVoice | None
    voices: VoiceConfig[OmniVoicePreset] | None
    voice_prompts: dict[str | None, VoiceClonePrompt | None]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.omnivoice_model, load_voice_config(settings, OmniVoicePreset))
        self.tts = None
        self.voice_prompts = {}

    @override
    def load(self) -> None:
        presets = self._reference_presets()
        device = self.settings.omnivoice_device.strip() or "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.tts = OmniVoice.from_pretrained(
            self.settings.omnivoice_model,
            device_map=device,
            dtype=dtype,
        )
        self.sample_rate = int(self.tts.sampling_rate)

        self.voice_prompts = {
            name: self._clone_prompt(self.tts, preset) for name, preset in presets.items()
        }
        self.loaded = True

    def _reference_presets(self) -> dict[str | None, OmniVoicePreset | None]:
        if self.voices is not None:
            return dict(self.voices.voices)
        audio = self.settings.omnivoice_ref_audio
        text = self.settings.omnivoice_ref_text
        if audio is None and text is None:
            return {None: None}
        if audio is None or text is None:
            raise ValueError(
                "TINYTALK_OMNIVOICE_REF_AUDIO and TINYTALK_OMNIVOICE_REF_TEXT "
                "must be configured together"
            )
        return {None: OmniVoicePreset(ref_audio=audio, ref_text=text)}

    @staticmethod
    def _clone_prompt(model: OmniVoice, preset: OmniVoicePreset | None) -> VoiceClonePrompt | None:
        if preset is None:
            return None
        transcript = preset.ref_text.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError(f"OmniVoice reference transcript is empty: {preset.ref_text}")
        return model.create_voice_clone_prompt(ref_audio=str(preset.ref_audio), ref_text=transcript)

    @override
    def prepare_request(
        self,
        *,
        instructions: str | None,
        speed: float | None,
        voice: str | None = None,
    ) -> SynthesisRequest:
        if speed is not None and speed <= 0:
            raise ValueError("speed must be positive")

        return SynthesisRequest(
            instructions=instructions.strip() if instructions else None, speed=speed, voice=voice
        )

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

        instructions = request.instructions
        prompt = self.voice_prompts[request.voice]
        speed = request.speed
        language = self.settings.omnivoice_language.strip() or None
        mode = self._mode(instructions, prompt)
        num_attempts = self.settings.max_retries + 1
        best_audio: AudioSamples | None = None
        best_quality: QualityResult | None = None
        best_attempt = -1
        best_temperature = 0.0
        wer_fallbacks = 0
        attempt_details: list[AttemptTiming] = []

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
                prompt=prompt,
                instructions=instructions,
                speed=speed,
                language=language,
                temperature=temperature,
            )
            wer_fallbacks += int(quality.wer_fallback)

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
        mode: VoiceMode,
        prompt: VoiceClonePrompt | None,
        instructions: str | None,
        speed: float | None,
        language: str | None,
        temperature: float,
    ) -> tuple[AudioSamples, QualityResult, AttemptTiming]:
        raw_audio, t_infer = self._generate_audio(
            chunk_text,
            mode=mode,
            prompt=prompt,
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

        detail = AttemptTiming(
            attempt=attempt,
            t_infer=t_infer,
            t_dsp=t_dsp,
            t_wer_check=t_wer_check,
            repeat_penalty=None,
            class_temperature=temperature,
            mode=mode,
            status=None,
            **quality.model_dump(),
        )
        return wav, quality, detail

    def _generate_audio(
        self,
        text: str,
        *,
        mode: VoiceMode,
        prompt: VoiceClonePrompt | None,
        instructions: str | None,
        language: str | None,
        speed: float | None,
        temperature: float,
    ) -> tuple[AudioArray, float]:
        if self.tts is None:
            raise RuntimeError("engine not loaded. Call load() first")

        start = time.perf_counter()
        audios = self.tts.generate(
            text=text,
            language=language,
            voice_clone_prompt=prompt if mode == "clone" else None,
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
        audio: AudioArray,
        *,
        index: int,
        num_chunks: int,
    ) -> tuple[AudioSamples, float]:
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
        wav: AudioSamples,
        chunk_text: str,
        *,
        mode: VoiceMode,
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
        mode: VoiceMode,
        audio: AudioSamples,
        quality: QualityResult,
        temperature: float,
        selected_attempt: int,
        details: list[AttemptTiming],
    ) -> ChunkTiming:
        status: AttemptStatus = (
            "accepted"
            if quality_is_acceptable(quality, self.settings.wer_threshold)
            else "fallback"
        )
        details[selected_attempt].status = status

        return ChunkTiming(
            index=index,
            attempts=len(details),
            status=status,
            duration=len(audio) / self.sample_rate,
            repeat_penalty=None,
            class_temperature=temperature,
            mode=mode,
            t_f0=0.0,
            attempts_detail=details,
            **quality.model_dump(),
        )

    @staticmethod
    def _mode(instructions: str | None, prompt: VoiceClonePrompt | None) -> VoiceMode:
        if instructions:
            return "design"
        if prompt is not None:
            return "clone"
        return "auto"

