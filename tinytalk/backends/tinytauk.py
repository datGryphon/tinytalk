from __future__ import annotations

import json
import time

import numpy as np
from tinytauk import TinyTAuK

from ..audio import edge_fade, peak_limit, silence, trim_edge_silence
from ..chunking import split_text
from ..config import Settings
from ..engine import RequestTiming, SynthesisResult

_DEFAULT_DESCRIPTION = "A clear, natural speaking voice"
_WARMUP_TEXT = (
    "TinyTalk is warming the speech runtime before serving requests so the first user synthesis "
    "runs on the compiled path."
)
_WARMUP_SECONDS = 9.0


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
        base_seed = self.tts.config.runtime.seed

        for index, chunk in enumerate(chunks):
            infer_start = time.perf_counter()
            result = self.tts.generate(
                self._instruction(chunk, instructions),
                gen_seconds=self._duration_seconds(chunk, speed_value),
                seed=base_seed + index,
            )
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
                trailing=index < len(chunks) - 1,
            )
            # Preserve AuK's requested loudness/prosody. Only guard clipping and
            # soften splice edges; NeuTTS-specific RMS/F0 normalization is not used.
            wav = peak_limit(wav)
            wav = edge_fade(wav, 3.0, self.sample_rate)
            t_dsp = time.perf_counter() - dsp_start

            if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                parts.append(
                    silence(
                        self.sample_rate,
                        self.settings.inter_chunk_silence_ms,
                        wav.dtype,
                    )
                )
            parts.append(wav)

            chunk_timings.append(
                {
                    "index": index,
                    "attempts": 1,
                    "duration": float(len(wav) / self.sample_rate),
                    "repeat_penalty": None,
                    "wer": None,
                    "wer_source": None,
                    "wer_fallback": False,
                    "t_f0": 0.0,
                    "attempts_detail": [
                        {
                            "attempt": 0,
                            "t_infer": t_infer,
                            "t_dsp": t_dsp,
                            "t_wer_check": 0.0,
                            "wer": None,
                            "wer_source": None,
                            "wer_fallback": False,
                            "repeat_penalty": None,
                            "accepted": True,
                        }
                    ],
                }
            )

        return SynthesisResult(
            audio=np.concatenate(parts) if len(parts) > 1 else parts[0],
            sample_rate=self.sample_rate,
            chunks=chunks,
            timing=RequestTiming(chunks=chunk_timings),
        )

    def _duration_seconds(self, text: str, speed: float) -> float:
        seconds = len(text) / self.settings.tinytauk_chars_per_second / speed
        return max(1.0, seconds)

    @staticmethod
    def _instruction(text: str, instructions: str | None) -> str:
        description = (
            instructions.strip()
            if instructions and instructions.strip()
            else _DEFAULT_DESCRIPTION
        )
        quoted_description = json.dumps(description, ensure_ascii=False)
        quoted_text = json.dumps(text, ensure_ascii=False)
        return (
            "Generate speech based on the following description: "
            f"{quoted_description}. The content to speak is: {quoted_text}."
        )
