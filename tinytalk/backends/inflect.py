"""Bounded Inflect Mini v3 adapter (stock voices, local profiles, events) owning only the
TinyTalk seam around the official release: a pinned release digest verified with the
package's own ``verify_release``, one resident events delivery, and a ``weights_only``
imported tuner profile loader. A configured profile takes precedence over the stock voice
and never renders event markup; all audio is native float32 24 kHz with no NeuTTS transforms.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..audio import silence
from ..chunking import split_text
from ..config import Settings
from ..engine import RequestTiming, SynthesisResult

RELEASE_MANIFEST_SHA256 = "f1f7ffa89905e024b08fcef82216b2901bbf6db6b00358b44cc4448abcbca7c6"
VOICES = ("hope", "sophia", "cassidy", "james", "jasper", "mark")
EVENT_MARKUP_RE = re.compile(r"\[[^\]\r\n]*\]|</?[A-Za-z][^>\r\n]*>")
# Opaque single-word guard for event markup: shared split_text treats it as one
# token, so it cannot re-tokenize or hard-wrap the surrounding bracket.
_EVENT_PLACEHOLDER_BASE = "ZZQZZPH"
PROFILE_POINTER_SCHEMA = "inflect-tuner-profile-pointer-v1"
NATIVE_SAMPLE_RATE = 24_000
BASE_SEED = 42


def _to_numpy(audio, sample_rate) -> np.ndarray:
    """Validate native audio and convert; no DSP. Fail on any non-canonical form."""
    if sample_rate != NATIVE_SAMPLE_RATE:
        raise RuntimeError(f"inflect audio must be {NATIVE_SAMPLE_RATE} Hz, got {sample_rate}")
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise RuntimeError("inflect audio must be mono")
    if audio.size == 0:
        raise RuntimeError("inflect audio is empty")
    audio = audio.astype(np.float32, copy=True)
    if not bool(np.all(np.isfinite(audio))):
        raise RuntimeError("inflect audio contains non-finite samples")
    if float(audio.max()) > 1.0 or float(audio.min()) < -1.0:
        raise RuntimeError("inflect audio samples are outside [-1, 1]")
    return audio


def _is_local_filename(name: str) -> bool:
    return (
        isinstance(name, str)
        and "/" not in name
        and "\\" not in name
        and name not in (".", "..")
        and ".." not in name.replace("\\", "/").split("/")
        and not Path(name).is_absolute()
    )


def _model_weights_sha(package: Path) -> str | None:
    """Read the release-verified export model manifest weights SHA256."""
    manifest_path = package / "core" / "baseline" / "models" / "mini" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("files", {}).get("weights.pt")


class InflectMiniEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded = False
        self.sample_rate = NATIVE_SAMPLE_RATE
        self.model_name = "inflect/mini-v3"
        self.adapter_loads = 0
        self._adapter = None
        self._delivery = None
        self._core = None
        self._frontend = None
        self._profile = None

    def load(self) -> None:
        if self.loaded:
            return
        package = Path(self.settings.inflect_package)
        self._verify_release(package)
        self._load_adapter(package)

        speech = self._adapter.base.speech
        self._delivery = speech.model
        self._core = speech.runtime[0]
        self._frontend = speech.runtime[1]

        if self.settings.inflect_profile is not None:
            self._load_profile()

        self.loaded = True

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        if instructions:
            raise ValueError("inflect does not accept instructions")
        if speed is not None and speed != 1.0:
            raise ValueError("inflect does not accept speed; keep speed at 1.0")
        if self._delivery is None:
            raise RuntimeError("engine not loaded. Call load() first")

        # Reject a configured profile combined with any event markup before generating
        # any chunk, so a rejected request never renders partial audio.
        profile = self._profile
        if profile is not None:
            voice = None
            if EVENT_MARKUP_RE.search(text):
                raise ValueError("conditioned profile with events is not supported")
        else:
            voice = self.settings.inflect_voice
            if voice not in VOICES:
                raise ValueError(f"unsupported inflect voice: {voice!r}")

        spans = self._protected_spans(text)
        parts: list[np.ndarray] = []
        chunk_timings: list[dict] = []
        for index, span in enumerate(spans):
            start = time.perf_counter()
            seed = (BASE_SEED + index) % (2**32)
            if EVENT_MARKUP_RE.search(span):
                result = self._adapter.synthesize(span, voice=voice, seed=seed, events=True)
                mode, extra_timing = "events", {}
            else:
                phonemes = self._text_to_phonemes(span)
                if profile is not None:
                    ids = self._delivery.base.phoneme_ids(phonemes)
                    result = profile.synthesize_ids(self._delivery, ids, seed=seed)
                    ntp = result.get("metadata", {}).get("native_tuner_profile", {})
                    if ntp.get("profile_applied") is not True or ntp.get("full_anchor_fallback") is not False:
                        raise RuntimeError("inflect profile did not apply cleanly")
                    mode, extra_timing = "profile", {"profile_applied": True, "full_anchor_fallback": False}
                else:
                    result = self._delivery.synthesize_phonemes(phonemes, voice=voice, seed=seed)
                    mode, extra_timing = "stock", {}
            audio = _to_numpy(result["audio"], result["sample_rate"])

            chunk_timings.append({
                "index": index,
                "attempts": 1,
                "t_infer": time.perf_counter() - start,
                "duration": len(audio) / self.sample_rate,
                "mode": mode,
                **extra_timing,
            })
            if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                parts.append(silence(self.sample_rate, self.settings.inter_chunk_silence_ms, audio.dtype))
            parts.append(audio)

        audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            chunks=list(spans),
            timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=0),
        )

    def _protected_spans(self, text: str) -> list[str]:
        """Split text preserving event markup verbatim.

        shared split_text re-tokenizes brackets and can hard-wrap them, so each tag is
        swapped for a placeholder absent from the original text across the split, then
        restored to the exact original token; markup the regex does not recognize is
        left untouched.
        """
        matches = list(EVENT_MARKUP_RE.finditer(text))
        if not matches:
            return split_text(text, self.settings.max_chars_per_chunk)

        placeholder_to_token: dict[str, str] = {}
        protected = text
        index = 0
        for match in matches:
            placeholder = _EVENT_PLACEHOLDER_BASE + str(index)
            index += 1
            while placeholder in protected:
                placeholder = _EVENT_PLACEHOLDER_BASE + str(index)
                index += 1
            placeholder_to_token[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)

        local_max = max(self.settings.max_chars_per_chunk, max(len(p) for p in placeholder_to_token))
        spans = split_text(protected, local_max)

        ordered = sorted(placeholder_to_token, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(p) for p in ordered))
        return [pattern.sub(lambda m: placeholder_to_token[m.group()], span) for span in spans]

    def _text_to_phonemes(self, span: str) -> str:
        prepared = self._core.prepare_phonemes(span, self._delivery.vocab, g2p=self._frontend)
        return prepared["ps"]

    def _verify_release(self, package: Path) -> None:
        # Bind the digest before running the package verifier, so a tampered
        # inventory refuses to execute any of its own code.
        actual = hashlib.sha256((package / "release-manifest.json").read_bytes()).hexdigest()
        if actual != RELEASE_MANIFEST_SHA256:
            raise ValueError("inflect release manifest does not match the pinned digest")
        spec = importlib.util.spec_from_file_location(
            "inflect_release_verify", package / "verify_release.py"
        )
        verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify)
        verify.verify(package)

    def _load_adapter(self, package: Path) -> None:
        spec = importlib.util.spec_from_file_location(
            "inflect_events_command", package / "extras" / "events" / "command.py"
        )
        command = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(command)
        self._adapter = command.load(str(package))
        self.adapter_loads += 1

    def _load_profile(self) -> None:
        companion = Path(self.settings.inflect_profile)
        pointer = json.loads(companion.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("schema") != PROFILE_POINTER_SCHEMA:
            raise ValueError("inflect profile pointer has an unrecognized schema")
        filename = pointer.get("file")
        if not _is_local_filename(filename):
            raise ValueError("inflect profile file must be a local companion filename")
        expected_sha = pointer.get("sha256")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("inflect profile sha256 must be 64 hex characters")

        profile_path = companion.parent / filename
        tuner_profile = self._load_tuner_profile_runtime()
        # Official loader verifies the SHA256, deserializes weights_only, and
        # validates the anchor; none of that is re-implemented here.
        profile = tuner_profile.load(profile_path, expected_sha)
        self._verify_profile_provenance(profile)
        self._profile = profile

    def _verify_profile_provenance(self, profile) -> None:
        """Bind the profile's claimed mini weights to the release-verified manifest.

        The upstream playback loader checks provenance syntax; the adapter binds the
        claimed weights to the manifest so a profile whose identity drifts from the
        verified export fails startup instead of serving silently.
        """
        claimed = profile.as_dict().get("provenance", {}).get("source_hashes", {}).get("mini_weights")
        if claimed != _model_weights_sha(self.settings.inflect_package):
            raise ValueError("inflect profile mini_weights do not match the release-verified manifest")

    def _load_tuner_profile_runtime(self) -> object:
        runtime_dir = (
            Path(self.settings.inflect_package)
            / "extras"
            / "tuner"
            / "playback"
            / "runtime"
        )
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))
        spec = importlib.util.spec_from_file_location(
            "inflect_tuner_profile", runtime_dir / "tuner_profile_20261110.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
