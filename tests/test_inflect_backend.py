"""Mocked Inflect Mini backend tests. Independent of any optional ML dependency:
the upstream adapter, resident delivery, core g2p, and tuner profile loader are
all replaced with fakes, and audio is plain float32 numpy.
"""
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.backends import inflect as inflect_backend
from tinytalk.backends.inflect import (
    PROFILE_POINTER_SCHEMA,
    _EVENT_PLACEHOLDER_BASE,
    InflectMiniEngine,
    InflectValidationError,
)
from tinytalk.config import Settings
from tinytalk.engine import SynthesisResult

# anchor_sid 0 -> "hope"; the alias that selects the profile is "bert".
ANCHOR_VOICE = "hope"
ALIAS = "bert"
# Value no longer checked by TinyTalk (the official loader validates the anchor);
# kept so the fake profile shape matches upstream.
FAKE_MINI_WEIGHTS = "0" * 64

ALPHA = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeNativeMini:
    def __init__(self, vocab):
        self.vocab = vocab

    def phoneme_ids(self, phonemes):
        return [0, *[self.vocab[c] for c in phonemes], 0]


class FakeDelivery:
    """Mimics MiniDelivery.synthesize_phonemes; records voice/seed per call."""

    def __init__(self, amplitude=0.1):
        self.base = FakeNativeMini(ALPHA)
        self.vocab = ALPHA
        self.calls = []
        self._amplitude = amplitude

    def synthesize_phonemes(self, phonemes, *, voice, seed):
        self.calls.append({"phonemes": phonemes, "voice": voice, "seed": seed})
        audio = np.full(240, self._amplitude, dtype=np.float32)
        return {
            "audio": audio,
            "sample_rate": 24_000,
            "metadata": {"delivery": {"selected": "original", "voice": voice}},
        }


class FakeCore:
    def prepare_phonemes(self, text, vocab, g2p=None):
        return {"ps": "".join(c for c in text.lower() if c.isalpha())}


class FakeAdapter:
    def __init__(self, delivery=None):
        self._delivery = delivery or FakeDelivery()
        self._core = FakeCore()
        speech = types.SimpleNamespace(
            model=self._delivery,
            runtime=(self._core, object(), self._delivery, object()),
            torch=object(),
        )
        self.base = types.SimpleNamespace(speech=speech)
        self.event_calls = []

    def synthesize(self, text, *, voice, seed, events):
        self.event_calls.append({"text": text, "voice": voice, "events": events})
        audio = np.full(480, 0.2, dtype=np.float32)
        return {
            "audio": audio,
            "sample_rate": 24_000,
            "metadata": {
                "events": {
                    "profiles": [{"tag": "yawn", "id": "cassidy-yawn-03", "voice": "cassidy"}],
                    "source_audio_at_inference": False,
                    "shared_decoder": True,
                    "joins": {
                        "sample_rate": 24_000,
                        "segments": [
                            {
                                "kind": "speech",
                                "tag": None,
                                "text": "wait. ",
                                "phones": "w",
                                "start": 0,
                                "end": 23400,
                                "raw_pcm_sha256": "deadbeef",
                                "trim": "removed head",
                            },
                            {
                                "kind": "event",
                                "tag": "yawn",
                                "text": None,
                                "start": 23400,
                                "end": 46440,
                                "raw_frames": 23040,
                            },
                        ],
                    },
                    "speech_metadata": [{"text": "wait. ", "phones": "w\u02c8A"}],
                }
            },
        }


class FakeProfile:
    def __init__(self, *, anchor_sid=0, applied=True, fallback=False):
        self._anchor_sid = anchor_sid
        self._applied = applied
        self._fallback = fallback
        self.calls = []

    def synthesize_ids(self, model, ids, seed=42):
        self.calls.append({"ids": ids, "seed": seed})
        return {
            "audio": np.full(300, 0.15, dtype=np.float32),
            "sample_rate": 24_000,
            "metadata": {"native_tuner_profile": {
                "profile_applied": self._applied,
                "full_anchor_fallback": self._fallback,
            }},
        }

    def as_dict(self):
        return {
            "anchor_sid": self._anchor_sid,
            "provenance": {"source_hashes": {"mini_weights": FAKE_MINI_WEIGHTS}},
        }


class _AbsentFieldProfile:
    """Fake profile returning caller-supplied tuner metadata, to exercise missing fields."""

    def __init__(self, metadata):
        self._metadata = metadata
        self.calls = []

    def synthesize_ids(self, model, ids, seed=42):
        self.calls.append({"ids": ids, "seed": seed})
        return {
            "audio": np.full(300, 0.15, dtype=np.float32),
            "sample_rate": 24_000,
            "metadata": self._metadata,
        }


class _ProvenanceProfile:
    """Fake profile exposing a controllable mini_weights provenance hash."""

    def __init__(self, *, mini_weights=FAKE_MINI_WEIGHTS):
        self._mini_weights = mini_weights
        self.calls = []

    def synthesize_ids(self, model, ids, seed=42):
        self.calls.append({"ids": ids, "seed": seed})
        return {"audio": np.full(300, 0.15, dtype=np.float32), "sample_rate": 24_000, "metadata": {}}

    def as_dict(self):
        source_hashes = {}
        if self._mini_weights is not None:
            source_hashes["mini_weights"] = self._mini_weights
        return {"anchor_sid": 0, "provenance": {"source_hashes": source_hashes}}


class FakeTunerProfileLoader:
    def __init__(self, *, profile=None, reject_sha=False):
        self._profile = profile or FakeProfile()
        self.reject_sha = reject_sha
        self.load_calls = []

    def load(self, path, expected_sha256):
        self.load_calls.append((str(path), expected_sha256))
        if self.reject_sha:
            raise ValueError("Profile size/SHA256 mismatch; refusing deserialization")
        return self._profile


def _patch_imported_module(monkeypatch, **namespace):
    """Return fake modules for the importlib.load pattern used by the backend."""

    class _Loader:
        def exec_module(self, module):
            pass

    def _fake_spec(*args, **kwargs):
        return types.SimpleNamespace(loader=_Loader())

    def _fake_module_from_spec(spec):
        module = types.ModuleType(getattr(spec, "name", "inflect_import"))
        module.__dict__.update(namespace)
        return module

    monkeypatch.setattr(importlib.util, "spec_from_file_location", _fake_spec)
    monkeypatch.setattr(importlib.util, "module_from_spec", _fake_module_from_spec)
    return namespace


def _stock_engine(profile=None, *, amplitude=0.1, inter_chunk_silence_ms=60,
                  max_chars_per_chunk=180, voice="cassidy", profile_voice=ALIAS):
    settings = Settings(
        backend="inflect",
        inflect_voice=voice,
        inflect_profile_voice=profile_voice,
        max_chars_per_chunk=max_chars_per_chunk,
        inter_chunk_silence_ms=inter_chunk_silence_ms,
    )
    engine = InflectMiniEngine(settings)
    delivery = FakeDelivery(amplitude=amplitude)
    engine._adapter = FakeAdapter(delivery)
    engine._delivery = delivery
    engine._core = FakeCore()
    engine._frontend = object()
    engine._profile = profile
    engine._profile_voice = profile_voice if profile is not None else None
    engine.loaded = True
    return engine


def _write_profile_files(tmp_path, *, sha="0" * 64):
    pt = tmp_path / "prof.pt"
    pt.write_bytes(b"profile-bytes")
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"file": "prof.pt", "schema": PROFILE_POINTER_SCHEMA, "sha256": sha}))
    return ptr, pt


# --------------------------------------------------------------------------- #
# Routing: alias selects profile; the six stock names (incl. anchor) are stock
# --------------------------------------------------------------------------- #
def test_alias_selects_profile_not_anchor():
    profile = FakeProfile(anchor_sid=0)  # anchor voice is "hope"
    engine = _stock_engine(profile=profile, voice=ANCHOR_VOICE)

    # The anchor voice renders stock even though a profile is loaded.
    stock = engine.synthesize("hello", voice=ANCHOR_VOICE)
    assert stock.timing.chunks[0]["mode"] == "stock"
    assert engine._profile.calls == []

    # The alias selects the cached profile; its internal anchor stays internal.
    chosen = engine.synthesize("hello", voice=ALIAS)
    assert chosen.timing.chunks[0]["mode"] == "profile"
    assert chosen.timing.chunks[0]["voice"] == ALIAS
    assert len(engine._profile.calls) == 1


@pytest.mark.parametrize("stock_voice", ["hope", "sophia", "cassidy", "james", "jasper", "mark"])
def test_all_stock_names_render_stock_with_profile_loaded(stock_voice):
    engine = _stock_engine(profile=FakeProfile(anchor_sid=0))
    result = engine.synthesize("hello world", voice=stock_voice)
    assert result.timing.chunks[0]["mode"] == "stock"
    assert result.timing.chunks[0]["voice"] == stock_voice
    assert engine._profile.calls == []


def test_unknown_alias_without_profile_is_rejected():
    engine = _stock_engine(profile=None)
    with pytest.raises(InflectValidationError, match="voice"):
        engine.synthesize("hello", voice=ALIAS)


# --------------------------------------------------------------------------- #
# Events: stock-voice events work with a profile loaded; alias+event is rejected
# --------------------------------------------------------------------------- #
def test_stock_event_works_with_profile_loaded():
    engine = _stock_engine(profile=FakeProfile(anchor_sid=0), voice="cassidy")
    result = engine.synthesize("wait. [yawn] what?", voice="cassidy")

    assert result.timing.chunks[0]["mode"] == "events"
    assert engine._adapter.event_calls[0]["voice"] == "cassidy"


def test_alias_event_is_rejected_as_conditioned():
    engine = _stock_engine(profile=FakeProfile(anchor_sid=0), voice="cassidy")
    with pytest.raises(InflectValidationError, match="event"):
        engine.synthesize("wait. [yawn] what?", voice=ALIAS)


def test_events_metadata_strips_input_text_and_phones():
    engine = _stock_engine(voice="cassidy")
    events = engine.synthesize("wait. [yawn] what was that?", voice="cassidy").timing.chunks[0]["events"]

    assert events is not None
    assert "speech_metadata" not in events
    assert events["profiles"][0]["tag"] == "yawn"  # tags preserved
    for seg in events["joins"]["segments"]:
        assert "text" not in seg
        assert "phones" not in seg
        assert "raw_pcm_sha256" not in seg
        assert "trim" not in seg
    # frame intervals retained
    assert events["joins"]["segments"][0]["start"] == 0
    assert events["joins"]["segments"][1]["raw_frames"] == 23040


def test_event_only_input_is_handled():
    engine = _stock_engine(voice="cassidy")
    result = engine.synthesize("[yawn]")
    assert result.timing.chunks[0]["mode"] == "events"


def test_native_parser_errors_are_preserved():
    engine = _stock_engine(voice="cassidy")

    class BrokenAdapter(FakeAdapter):
        def synthesize(self, text, **kwargs):
            raise ValueError("Unknown event tag")

    engine._adapter = BrokenAdapter()
    with pytest.raises(ValueError, match="Unknown event tag"):
        engine.synthesize("hi [bogus] there", voice="cassidy")


# --------------------------------------------------------------------------- #
# _event_spans: shared split_text must preserve original event markup verbatim
# --------------------------------------------------------------------------- #
def _event_engine(max_chars_per_chunk: int = 180) -> InflectMiniEngine:
    return InflectMiniEngine(
        Settings(
            backend="inflect",
            inflect_voice="cassidy",
            inflect_profile_voice="bert",
            max_chars_per_chunk=max_chars_per_chunk,
        )
    )


def test_event_spans_preserve_valid_tag_across_sentence_boundary():
    engine = _event_engine()
    text = "That was surprising. [laugh] Let us continue."

    spans = engine._event_spans(text)

    assert " ".join(spans) == text  # split_text alone corrupts this to "[ laugh]"
    assert any("[laugh]" in span for span in spans)


def test_event_spans_leaves_invalid_tag_untouched():
    engine = _event_engine()
    text = "That was surprising. [ laugh] Let us continue."

    spans = engine._event_spans(text)
    joined = " ".join(spans)

    assert joined == text  # invalid markup preserved verbatim
    assert "[ laugh]" in joined
    assert "[laugh]" not in joined  # never silently repaired to a valid tag


def test_event_spans_multitag_and_small_limit_not_corrupted():
    engine = _event_engine(max_chars_per_chunk=40)
    text = "Hold on. [cough] Wait. [gasp] What now?"

    joined = " ".join(engine._event_spans(text))

    assert "[cough]" in joined
    assert "[gasp]" in joined
    assert "[ cough]" not in joined and "[ gasp]" not in joined


def test_event_spans_placeholder_literal_collision_preserves_user_text():
    engine = _event_engine()
    collision = f"{_EVENT_PLACEHOLDER_BASE}0"
    text = f"user note {collision} here [laugh] ok"

    joined = " ".join(engine._event_spans(text))

    assert joined == text  # user placeholder literal untouched, tag restored
    assert collision in joined
    assert "[laugh]" in joined


# --------------------------------------------------------------------------- #
# Profile conditioning metadata must apply cleanly or raise
# --------------------------------------------------------------------------- #
def test_profile_bad_metadata_raises_runtime_error():
    engine = _stock_engine(profile=FakeProfile(applied=False, fallback=True), voice=ANCHOR_VOICE)
    with pytest.raises(RuntimeError, match="did not apply"):
        engine.synthesize("hello", voice=ALIAS)


def test_profile_full_anchor_fallback_raises_runtime_error():
    engine = _stock_engine(profile=FakeProfile(fallback=True), voice=ANCHOR_VOICE)
    with pytest.raises(RuntimeError):
        engine.synthesize("hello", voice=ALIAS)


def test_profile_mode_uses_profile_synthesize_ids():
    profile = FakeProfile(anchor_sid=0, applied=True, fallback=False)
    engine = _stock_engine(profile=profile, voice=ANCHOR_VOICE)

    result = engine.synthesize("hello world", voice=ALIAS)

    assert len(engine._delivery.calls) == 0  # stock path bypassed
    assert len(profile.calls) == 1
    call = profile.calls[0]
    assert call["ids"][0] == 0 and call["ids"][-1] == 0  # boundary-wrapped ids
    timing = result.timing.chunks[0]
    assert timing["mode"] == "profile"
    assert timing["attempts"] == 1
    assert timing["profile_applied"] is True
    assert timing["full_anchor_fallback"] is False
    assert timing["voice"] == ALIAS


@pytest.mark.parametrize("metadata", [
    {},  # missing native_tuner_profile entirely
    {"native_tuner_profile": {"full_anchor_fallback": False}},  # missing profile_applied
    {"native_tuner_profile": {"profile_applied": True}},  # missing full_anchor_fallback
])
def test_profile_missing_metadata_raises_runtime_error(metadata):
    engine = _stock_engine(profile=_AbsentFieldProfile(metadata), voice=ANCHOR_VOICE)
    with pytest.raises(RuntimeError):
        engine.synthesize("hello", voice=ALIAS)


# --------------------------------------------------------------------------- #
# _to_numpy: enforce native 24k nonempty mono finite float [-1, 1]
# --------------------------------------------------------------------------- #
def test_to_numpy_accepts_native_mono():
    out = inflect_backend._to_numpy(np.full(100, 0.5, dtype=np.float32), 24_000)
    assert out.dtype == np.float32 and out.ndim == 1 and out.size == 100


@pytest.mark.parametrize("audio", [
    np.full(100, 0.1, dtype=np.float32),
    np.full((2, 100), 0.1, dtype=np.float32),
    np.array([], dtype=np.float32),
])
def test_to_numpy_rejects_bad_shape_or_rate(audio):
    with pytest.raises(RuntimeError):
        inflect_backend._to_numpy(audio, 16_000 if audio.ndim == 1 else 24_000)


def test_to_numpy_rejects_empty_mono():
    with pytest.raises(RuntimeError):
        inflect_backend._to_numpy(np.array([], dtype=np.float32), 24_000)


def test_to_numpy_rejects_nonfinite():
    audio = np.full(100, 0.1, dtype=np.float32)
    audio[10] = float("nan")
    with pytest.raises(RuntimeError):
        inflect_backend._to_numpy(audio, 24_000)


def test_to_numpy_rejects_out_of_range():
    with pytest.raises(RuntimeError):
        inflect_backend._to_numpy(np.full(100, 1.5, dtype=np.float32), 24_000)


# --------------------------------------------------------------------------- #
# Profile pointer validation: local filename, sha format, alias
# --------------------------------------------------------------------------- #
def test_profile_bad_pointer_schema_is_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": "wrong", "file": "prof.pt", "sha256": "0" * 64}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    with pytest.raises(InflectValidationError, match="schema"):
        engine._load_profile()
    assert loader.load_calls == []


def test_profile_file_traversal_is_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": PROFILE_POINTER_SCHEMA, "file": "../evil.pt", "sha256": "0" * 64}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    with pytest.raises(InflectValidationError, match="local companion"):
        engine._load_profile()


def test_profile_absolute_file_is_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": PROFILE_POINTER_SCHEMA, "file": "/tmp/evil.pt", "sha256": "0" * 64}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    with pytest.raises(InflectValidationError, match="local companion"):
        engine._load_profile()


def test_profile_bad_sha_format_is_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": PROFILE_POINTER_SCHEMA, "file": "prof.pt", "sha256": "not-a-sha"}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    with pytest.raises(InflectValidationError, match="64 hex"):
        engine._load_profile()


def test_profile_empty_alias_is_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=""))
    with pytest.raises(InflectValidationError, match="empty"):
        engine._load_profile()


def test_profile_reserved_alias_is_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice="cassidy"))
    with pytest.raises(InflectValidationError, match="reserved"):
        engine._load_profile()


def test_profile_loads_from_local_companion_directory(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: FAKE_MINI_WEIGHTS)
    ptr, pt = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    engine._verify_release = lambda package: None
    engine._load_adapter = lambda package: setattr(engine, "_adapter", FakeAdapter())
    engine.load()

    assert len(loader.load_calls) == 1
    assert Path(loader.load_calls[0][0]).parent == tmp_path  # resolved beside the pointer
    assert engine._profile is loader._profile
    assert engine._profile_voice == ALIAS  # alias, not the anchor


# --------------------------------------------------------------------------- #
# Release bind + load lifecycle
# --------------------------------------------------------------------------- #
def test_verify_release_checks_digest_before_verifier(monkeypatch, tmp_path):
    ran = {"verify": False}

    def fake_verify(package):
        ran["verify"] = True

    _patch_imported_module(monkeypatch, verify=fake_verify, load=lambda *a, **k: FakeAdapter())
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({"schema": "x", "files": {}}))  # digest will not match

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_package=tmp_path))
    with pytest.raises(InflectValidationError, match="pinned digest"):
        engine.load()
    assert ran["verify"] is False  # the package verifier never executed


def test_load_verifies_release_and_builds_delivery_once(monkeypatch, tmp_path):
    calls = {"verify": 0, "adapter": 0}

    def fake_verify(package):
        calls["verify"] += 1

    def fake_load(package_root, **kwargs):
        calls["adapter"] += 1
        return FakeAdapter()

    _patch_imported_module(monkeypatch, verify=fake_verify, load=fake_load)
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({"schema": "inflect-v3", "files": {}}))
    monkeypatch.setattr(
        inflect_backend,
        "RELEASE_MANIFEST_SHA256",
        __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
    )

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_package=tmp_path))
    engine.load()

    assert calls == {"verify": 1, "adapter": 1}
    assert engine.loaded is True
    assert engine.adapter_loads == 1
    assert engine._delivery is not None


def test_load_is_idempotent(monkeypatch):
    engine = InflectMiniEngine(Settings(backend="inflect"))
    calls = {"verify": 0, "adapter": 0}

    def count_verify(package):
        calls["verify"] += 1

    def count_adapter(package):
        calls["adapter"] += 1
        engine._adapter = FakeAdapter()

    engine._verify_release = count_verify
    engine._load_adapter = count_adapter

    engine.load()
    engine.load()

    assert calls == {"verify": 1, "adapter": 1}
    assert engine.loaded is True


def test_profile_is_loaded_once_and_cached(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: FAKE_MINI_WEIGHTS)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(
        Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS)
    )
    engine._verify_release = lambda package: None
    engine._load_adapter = lambda package: setattr(engine, "_adapter", FakeAdapter())
    engine.load()
    assert len(loader.load_calls) == 1

    engine.synthesize("hello world", voice=ALIAS)
    engine.synthesize("again now", voice=ALIAS)
    assert len(loader.load_calls) == 1
    assert len(engine._profile.calls) == 2


# --------------------------------------------------------------------------- #
# Profile provenance bound to the release-verified manifest weights
# --------------------------------------------------------------------------- #
def test_profile_provenance_matches_manifest_passes(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader(profile=_ProvenanceProfile(mini_weights=FAKE_MINI_WEIGHTS))
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: FAKE_MINI_WEIGHTS)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    engine._load_profile()

    assert engine._profile is loader._profile
    assert engine._profile_voice == ALIAS


def test_profile_provenance_mismatch_fails(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader(profile=_ProvenanceProfile(mini_weights=FAKE_MINI_WEIGHTS))
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: "1" * 64)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    with pytest.raises(InflectValidationError, match="manifest"):
        engine._load_profile()
    assert engine._profile is None


def test_profile_provenance_missing_source_hash_fails(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader(profile=_ProvenanceProfile(mini_weights=None))
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: FAKE_MINI_WEIGHTS)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr, inflect_profile_voice=ALIAS))
    with pytest.raises(InflectValidationError, match="provenance"):
        engine._load_profile()
    assert engine._profile is None


# --------------------------------------------------------------------------- #
# Stock synthesis timing + assembly
# --------------------------------------------------------------------------- #
def test_stock_uses_delivery_phoneme_path_and_reuses_delivery():
    engine = _stock_engine(max_chars_per_chunk=20)
    result = engine.synthesize(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    )

    assert result.sample_rate == 24_000
    assert len(engine._delivery.calls) >= 2  # one delivery call per split span
    for call in engine._delivery.calls:
        assert call["voice"] == "cassidy"
    timing = result.timing.chunks
    assert all(chunk["attempts"] == 1 for chunk in timing)
    assert all(chunk["mode"] == "stock" for chunk in timing)


def test_chunk_timing_has_duration_and_inference_elapsed():
    engine = _stock_engine(max_chars_per_chunk=180)
    chunk = engine.synthesize("hello world", voice="cassidy").timing.chunks[0]

    assert chunk["duration"] == pytest.approx(len(engine.synthesize("hello world", voice="cassidy").audio) / 24_000)
    assert isinstance(chunk["t_infer"], (int, float))
    assert chunk["t_infer"] >= 0


def test_stock_adds_inter_chunk_silence_between_spans():
    engine = _stock_engine(inter_chunk_silence_ms=60, max_chars_per_chunk=20)
    result = engine.synthesize(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    )

    silence_samples = 24_000 * (60 / 1000)
    speech_samples = sum(240 for _ in engine._delivery.calls)
    assert len(result.audio) == speech_samples + int(silence_samples) * (len(engine._delivery.calls) - 1)


def test_stock_does_not_apply_neutts_dsp():
    engine = _stock_engine(amplitude=0.9)
    result = engine.synthesize("hello world", voice="cassidy")
    assert float(np.max(np.abs(result.audio))) == pytest.approx(0.9)  # passed through untouched


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_instructions_are_rejected():
    engine = _stock_engine()
    with pytest.raises(InflectValidationError, match="instructions"):
        engine.synthesize("hello world", instructions="speak calmly")


def test_speed_rejected_except_one():
    engine = _stock_engine()
    with pytest.raises(InflectValidationError, match="speed"):
        engine.synthesize("hello world", speed=1.5)
    engine.synthesize("hello world", speed=1.0)  # accepted
    engine.synthesize("hello world")  # omitted, accepted


def test_unknown_voice_is_rejected():
    engine = _stock_engine()
    with pytest.raises(InflectValidationError, match="voice"):
        engine.synthesize("hello world", voice="ghost")


def test_default_voice_is_used_when_omitted():
    settings = Settings(backend="inflect", inflect_voice="james")
    engine = InflectMiniEngine(settings)
    engine.loaded = True
    engine._adapter = FakeAdapter()
    engine._delivery = FakeDelivery()
    engine._core = FakeCore()
    engine._frontend = object()
    engine._profile = None

    engine.synthesize("hello world")
    assert engine._delivery.calls[-1]["voice"] == "james"


def test_not_loaded_raises():
    engine = InflectMiniEngine(Settings(backend="inflect"))
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.synthesize("hello world")


# --------------------------------------------------------------------------- #
# Server routing / 400 mapping / readiness
# --------------------------------------------------------------------------- #
def _server_engine(profile=None, *, voice="cassidy"):
    engine = InflectMiniEngine(
        Settings(
            backend="inflect",
            inflect_voice=voice,
            inflect_profile_voice=ALIAS,
            inflect_profile=profile and Path("/x"),
            inter_chunk_silence_ms=0,
        )
    )
    engine.loaded = True
    engine.load = lambda: None
    engine._adapter = FakeAdapter()
    engine._delivery = FakeDelivery()
    engine._core = FakeCore()
    engine._frontend = object()
    engine._profile = profile
    engine._profile_voice = ALIAS if profile is not None else None
    return engine


def test_unsupported_voice_returns_400(monkeypatch):
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", _server_engine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "hi", "voice": "ghost"})
    assert res.status_code == 400


def test_alias_event_returns_400(monkeypatch):
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", _server_engine(profile=FakeProfile(anchor_sid=0)))
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={"input": "wait. [yawn] what?", "voice": ALIAS},
        )
    assert res.status_code == 400


def test_instructions_return_400(monkeypatch):
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", _server_engine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "hi", "instructions": "speak"})
    assert res.status_code == 400


def test_speed_not_one_returns_400(monkeypatch):
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", _server_engine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "hi", "speed": 1.5})
    assert res.status_code == 400


def test_voice_is_forwarded_to_inflect_engine(monkeypatch):
    engine = _server_engine()
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech", json={"input": "hello world", "voice": ANCHOR_VOICE}
        )
    assert res.status_code == 200
    assert res.headers["X-TinyTalk-Backend"] == "inflect"
    assert engine._delivery.calls[-1]["voice"] == ANCHOR_VOICE


def test_other_backends_do_not_receive_voice(monkeypatch):
    calls = {}

    class PlainEngine:
        loaded = True
        model_name = "plain"

        def load(self):
            self.loaded = True

        def synthesize(self, text, *, instructions=None, speed=None, voice=None):
            calls.update(instructions=instructions, speed=speed, voice=voice)
            return SynthesisResult(
                audio=np.zeros(2400, dtype=np.float32),
                sample_rate=24_000,
                chunks=[text],
            )

    monkeypatch.setattr(server, "settings", Settings(backend="neutts"))
    monkeypatch.setattr(server, "engine", PlainEngine())
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "hope", "instructions": "x", "speed": 1.2},
        )
    assert res.status_code == 200
    # The neutts path forwards neither instructions, speed, nor voice.
    assert calls == {"instructions": None, "speed": None, "voice": None}


def test_health_reflects_readiness(monkeypatch):
    engine = InflectMiniEngine(Settings(backend="inflect"))
    engine.loaded = False
    engine.load = lambda: None
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 503
    engine.loaded = True
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
