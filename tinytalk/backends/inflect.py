"""Bounded Inflect Mini v3 adapter: stock voices, local tuner profiles, events.

Owns only the TinyTalk seam around the official release. The release inventory
is bound to a pinned manifest digest and then verified with the package's own
``verify_release``; the events ``command.load`` builds one resident
``MiniDelivery`` from official ``core`` paths; and the tuner profile loader is
imported from the official playback runtime and deserialized ``weights_only``
after a caller-supplied SHA256. All audio is native float32 24 kHz with no
NeuTTS transforms or quality retries.

A loaded tuner profile is selected by an external alias
(``inflect_profile_voice``); the six stock names (including the profile's anchor)
always render stock, and events render stock-voice only.
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


def _event_placeholder(index: int) -> str:
    return f"{_EVENT_PLACEHOLDER_BASE}{index}"


def _protect_placeholders(count: int, text: str) -> list[str]:
    """Pick `count` placeholders absent from user text, so restore cannot touch it."""
    placeholders: list[str] = []
    index = 0
    while len(placeholders) < count:
        candidate = _event_placeholder(index)
        if candidate not in text:
            placeholders.append(candidate)
        index += 1
    return placeholders
PROFILE_POINTER_SCHEMA = "inflect-tuner-profile-pointer-v1"
NATIVE_SAMPLE_RATE = 24_000
BASE_SEED = 42


class InflectValidationError(ValueError):
    """Bad inflect request mapped to HTTP 400 without hiding inference errors."""


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


def _has_dangling_tag(text: str) -> bool:
    return text.count("[") != text.count("]")


def _strip_event_metadata(events):
    """Drop input text/phones from event metadata; keep tags, frame intervals, flags."""
    if not isinstance(events, dict):
        return None
    joins = events.get("joins")
    segments = []
    for seg in joins.get("segments") if isinstance(joins, dict) else []:
        if not isinstance(seg, dict):
            continue
        segments.append(
            {
                "kind": seg.get("kind"),
                "tag": seg.get("tag"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "raw_frames": seg.get("raw_frames"),
                "retained_source_frames": seg.get("retained_source_frames"),
            }
        )
    flags = {
        key: events[key]
        for key in (
            "source_audio_at_inference",
            "shared_decoder",
            "owner_release_authorized",
            "diagnostic_candidate",
            "event_level_calibrated",
            "compressed_exemplars",
            "speech_weights_changed",
        )
        if key in events
    }
    return {
        "profiles": [
            {"tag": p.get("tag"), "id": p.get("id"), "voice": p.get("voice")}
            for p in events.get("profiles", [])
            if isinstance(p, dict)
        ],
        "joins": {
            "sample_rate": joins.get("sample_rate") if isinstance(joins, dict) else None,
            "segments": segments,
        },
        **flags,
    }


def _model_weights_sha(package: Path) -> str:
    """Read the release-verified export model manifest weights SHA256."""
    manifest_path = package / "core" / "baseline" / "models" / "mini" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_sha = manifest.get("files", {}).get("weights.pt")
    if not isinstance(weights_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", weights_sha):
        raise InflectValidationError("inflect model manifest missing valid weights.pt sha256")
    return weights_sha


class InflectMiniEngine:
    settings: Settings
    loaded: bool
    model_name: str
    sample_rate: int

    _adapter: object | None
    _delivery: object | None
    _core: object | None
    _frontend: object | None
    _profile: object | None
    # The external alias that selects the cached profile; never the internal anchor.
    _profile_voice: str | None
    # The events command is imported once and reused for stock/profile/event synthesis.
    adapter_loads: int

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
        self._profile_voice = None

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
        voice: str | None = None,
    ) -> SynthesisResult:
        if instructions:
            raise InflectValidationError("inflect does not accept instructions")
        if speed is not None and speed != 1.0:
            raise InflectValidationError("inflect does not accept speed; keep speed at 1.0")
        if self._delivery is None:
            raise RuntimeError("engine not loaded. Call load() first")

        stock_voice, profile_selected = self._resolve(voice)
        if EVENT_MARKUP_RE.search(text):
            if profile_selected:
                raise InflectValidationError("conditioned profile with events is not supported")
            return self._synthesize_events(text, stock_voice)
        if profile_selected:
            return self._synthesize_profile_text(text)
        return self._synthesize_stock(text, stock_voice)

    def _resolve(self, voice):
        """Map the request to a stock voice, or signal that the alias selects a profile."""
        requested = voice or self.settings.inflect_voice
        if requested in VOICES:
            return requested, False
        if self._profile is not None and requested == self._profile_voice:
            return None, True
        raise InflectValidationError(f"unsupported inflect voice: {requested!r}")

    def _verify_release(self, package: Path) -> None:
        # Bind the digest before running the package verifier, so a tampered
        # inventory refuses to execute any of its own code.
        actual = hashlib.sha256((package / "release-manifest.json").read_bytes()).hexdigest()
        if actual != RELEASE_MANIFEST_SHA256:
            raise InflectValidationError("inflect release manifest does not match the pinned digest")
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
            raise InflectValidationError("inflect profile pointer has an unrecognized schema")
        filename = pointer.get("file")
        expected_sha = pointer.get("sha256")
        if not _is_local_filename(filename):
            raise InflectValidationError("inflect profile file must be a local companion filename")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise InflectValidationError("inflect profile sha256 must be 64 hex characters")
        if not self.settings.inflect_profile_voice:
            raise InflectValidationError("inflect profile alias must not be empty")
        if self.settings.inflect_profile_voice in VOICES:
            raise InflectValidationError("inflect profile alias collides with a reserved stock voice name")

        profile_path = companion.parent / filename
        tuner_profile = self._load_tuner_profile_runtime()
        # Official loader verifies the SHA256, deserializes weights_only, and
        # validates the anchor; none of that is re-implemented here.
        profile = tuner_profile.load(profile_path, expected_sha)
        self._verify_profile_provenance(profile)
        self._profile = profile
        self._profile_voice = self.settings.inflect_profile_voice

    def _verify_profile_provenance(self, profile) -> None:
        """Bind the profile's claimed mini weights to the release-verified manifest.

        The upstream playback loader checks provenance syntax; the adapter binds
        the claimed weights to the manifest so a profile whose identity drifts
        from the verified export fails startup instead of serving silently.
        """
        provenance = profile.as_dict().get("provenance", {})
        claimed = provenance.get("source_hashes", {}).get("mini_weights")
        if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
            raise InflectValidationError("inflect profile missing valid mini_weights provenance")
        if claimed != _model_weights_sha(self.settings.inflect_package):
            raise InflectValidationError("inflect profile mini_weights do not match the release-verified manifest")

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

    def _synthesize_events(self, text: str, voice: str) -> SynthesisResult:
        spans = self._event_spans(text)
        parts: list[np.ndarray] = []
        chunk_timings: list[dict] = []
        for index, span in enumerate(spans):
            start = time.perf_counter()
            result = self._adapter.synthesize(span, voice=voice, seed=BASE_SEED, events=True)
            audio = _to_numpy(result["audio"], result["sample_rate"])
            chunk_timings.append(
                {
                    "index": index,
                    "attempts": 1,
                    "t_infer": time.perf_counter() - start,
                    "duration": len(audio) / self.sample_rate,
                    "mode": "events",
                    "voice": voice,
                    "events": _strip_event_metadata(result["metadata"].get("events")),
                }
            )
            if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                parts.append(
                    silence(self.sample_rate, self.settings.inter_chunk_silence_ms, audio.dtype)
                )
            parts.append(audio)
        audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            chunks=list(spans),
            timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=0),
        )

    def _event_spans(self, text: str) -> list[str]:
        """Return split spans with original event markup preserved verbatim.

        shared split_text re-tokenizes brackets at sentence boundaries and can
        hard-wrap them; guard each tag with an opaque placeholder across the
        split, then restore the exact original token so markup is never altered.
        """
        matches = list(EVENT_MARKUP_RE.finditer(text))
        if not matches:
            return split_text(text, self.settings.max_chars_per_chunk)

        placeholders = _protect_placeholders(len(matches), text)
        placeholder_to_token: dict[str, str] = {}
        protected = text
        for match, placeholder in zip(matches, placeholders):
            placeholder_to_token[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)

        local_max = max(self.settings.max_chars_per_chunk, max(len(p) for p in placeholders))
        spans = split_text(protected, local_max)

        ordered = sorted(placeholder_to_token, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(p) for p in ordered))
        return [pattern.sub(lambda m: placeholder_to_token[m.group()], span) for span in spans]

    def _synthesize_stock(self, text: str, voice: str) -> SynthesisResult:
        return self._assemble_spans(text, voice, self._stock_span)

    def _synthesize_profile_text(self, text: str) -> SynthesisResult:
        return self._assemble_spans(text, None, self._profile_span)

    def _assemble_spans(self, text: str, voice: str | None, span_fn) -> SynthesisResult:
        spans = split_text(text, self.settings.max_chars_per_chunk)
        parts: list[np.ndarray] = []
        chunk_timings: list[dict] = []
        for index, span in enumerate(spans):
            audio, meta = span_fn(span, (BASE_SEED + index) % (2**32), voice)
            chunk_timings.append(
                {
                    "index": index,
                    "attempts": 1,
                    "t_infer": meta.get("t_infer"),
                    "duration": meta.get("duration", len(audio) / self.sample_rate),
                    **meta["timing"],
                }
            )
            if index > 0 and self.settings.inter_chunk_silence_ms > 0:
                parts.append(
                    silence(self.sample_rate, self.settings.inter_chunk_silence_ms, audio.dtype)
                )
            parts.append(audio)
        audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            chunks=list(spans),
            timing=RequestTiming(chunks=chunk_timings, wer_fallbacks=0),
        )

    def _stock_span(self, span: str, seed: int, voice: str) -> tuple[np.ndarray, dict]:
        start = time.perf_counter()
        phonemes = self._text_to_phonemes(span)
        result = self._delivery.synthesize_phonemes(phonemes, voice=voice, seed=seed)
        audio = _to_numpy(result["audio"], result["sample_rate"])
        return audio, {
            "t_infer": time.perf_counter() - start,
            "duration": len(audio) / self.sample_rate,
            "timing": {
                "mode": "stock",
                "voice": voice,
                "delivery": result.get("metadata", {}).get("delivery"),
            },
        }

    def _profile_span(self, span: str, seed: int, voice: str | None) -> tuple[np.ndarray, dict]:
        start = time.perf_counter()
        phonemes = self._text_to_phonemes(span)
        ids = self._delivery.base.phoneme_ids(phonemes)
        result = self._profile.synthesize_ids(self._delivery, ids, seed=seed)
        ntp = result.get("metadata", {}).get("native_tuner_profile", {})
        if ntp.get("profile_applied") is not True or ntp.get("full_anchor_fallback") is not False:
            raise RuntimeError("inflect profile did not apply cleanly")
        audio = _to_numpy(result["audio"], result["sample_rate"])
        return audio, {
            "t_infer": time.perf_counter() - start,
            "duration": len(audio) / self.sample_rate,
            "timing": {
                "mode": "profile",
                "profile_applied": True,
                "full_anchor_fallback": False,
                "voice": self._profile_voice,
            },
        }

    def _text_to_phonemes(self, span: str) -> str:
        prepared = self._core.prepare_phonemes(span, self._delivery.vocab, g2p=self._frontend)
        return prepared["ps"]
