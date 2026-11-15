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
)
from tinytalk.config import Settings
from tinytalk.engine import SynthesisResult

# Posted HTTP voice that must be ignored; the configured stock voice is "cassidy".
ANCHOR_VOICE = "hope"
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
        return {
            "audio": np.full(480, 0.2, dtype=np.float32),
            "sample_rate": 24_000,
            "metadata": {},
        }


class FakeProfile:
    def __init__(self, *, applied=True, fallback=False):
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
            "anchor_sid": 0,
            "provenance": {"source_hashes": {"mini_weights": FAKE_MINI_WEIGHTS}},
        }


class _AbsentFieldProfile:
    """Fake profile returning caller-supplied tuner metadata, to exercise missing flags."""

    def __init__(self, metadata):
        self._metadata = metadata
        self.calls = []

    def synthesize_ids(self, model, ids, seed=42):
        self.calls.append({"ids": ids, "seed": seed})
        return {"audio": np.full(300, 0.15, dtype=np.float32), "sample_rate": 24_000, "metadata": self._metadata}


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
                  max_chars_per_chunk=180, voice="cassidy"):
    settings = Settings(
        backend="inflect",
        inflect_voice=voice,
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
    engine.loaded = True
    return engine


def _write_profile_files(tmp_path, *, sha="0" * 64):
    pt = tmp_path / "prof.pt"
    pt.write_bytes(b"profile-bytes")
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"file": "prof.pt", "schema": PROFILE_POINTER_SCHEMA, "sha256": sha}))
    return ptr, pt


# --------------------------------------------------------------------------- #
# Resolution: configured profile precedence, stock voice, events
# --------------------------------------------------------------------------- #
def test_configured_profile_takes_precedence_over_stock_voice():
    profile = FakeProfile()
    engine = _stock_engine(profile=profile, voice="cassidy")

    result = engine.synthesize("hello world")

    assert result.timing.chunks[0]["mode"] == "profile"
    assert len(profile.calls) == 1  # profile rendered, stock bypassed


def test_configured_stock_voice_is_used_without_profile():
    engine = _stock_engine(voice="james")

    result = engine.synthesize("hello world")

    assert result.timing.chunks[0]["mode"] == "stock"
    assert engine._delivery.calls[-1]["voice"] == "james"


def test_unconfigured_invalid_voice_rejected():
    engine = _stock_engine(voice="ghost")

    with pytest.raises(ValueError, match="voice"):
        engine.synthesize("hello world")


def test_conditioned_profile_with_events_is_rejected():
    engine = _stock_engine(profile=FakeProfile(), voice="cassidy")

    with pytest.raises(ValueError, match="event"):
        engine.synthesize("wait. [yawn] what?")


def test_event_chunk_uses_adapter_without_profile():
    engine = _stock_engine(voice="cassidy")

    result = engine.synthesize("wait. [yawn] what?")

    assert result.timing.chunks[0]["mode"] == "events"
    assert engine._adapter.event_calls[0]["voice"] == "cassidy"


def test_event_only_input_uses_adapter():
    engine = _stock_engine(voice="cassidy")

    result = engine.synthesize("[yawn]")

    assert result.timing.chunks[0]["mode"] == "events"


# --------------------------------------------------------------------------- #
# Tag protection: split must preserve event markup verbatim
# --------------------------------------------------------------------------- #
def test_tag_preserved_across_sentence_boundary():
    engine = _stock_engine(voice="cassidy")
    text = "That was surprising. [laugh] Let us continue."

    joined = " ".join(engine.synthesize(text).chunks)

    assert joined == text
    assert "[laugh]" in joined


def test_repeated_tags_preserved():
    engine = _stock_engine(voice="cassidy")
    text = "Hi. [yawn] there. [yawn] again."

    joined = " ".join(engine.synthesize(text).chunks)

    assert joined == text
    assert joined.count("[yawn]") == 2


def test_literal_collision_preserved():
    engine = _stock_engine(voice="cassidy")
    collision = f"{_EVENT_PLACEHOLDER_BASE}0"
    text = f"user note {collision} here [laugh] ok"

    joined = " ".join(engine.synthesize(text).chunks)

    assert joined == text  # user placeholder literal untouched, tag restored
    assert collision in joined
    assert "[laugh]" in joined


def test_small_chunk_limit_preserves_tags():
    engine = _stock_engine(voice="cassidy", max_chars_per_chunk=40)
    text = "Hold on. [cough] Wait. [gasp] What now?"

    joined = " ".join(engine.synthesize(text).chunks)

    assert "[cough]" in joined
    assert "[gasp]" in joined
    assert "[ cough]" not in joined and "[ gasp]" not in joined


# --------------------------------------------------------------------------- #
# Profile conditioning metadata must apply cleanly or raise
# --------------------------------------------------------------------------- #
def test_profile_fallback_failure_raises():
    engine = _stock_engine(profile=FakeProfile(fallback=True), voice="cassidy")

    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


def test_profile_missing_flags_raises():
    engine = _stock_engine(profile=_AbsentFieldProfile({"native_tuner_profile": {}}), voice="cassidy")

    with pytest.raises(RuntimeError):
        engine.synthesize("hello")


# --------------------------------------------------------------------------- #
# _to_numpy: enforce native 24k nonempty mono finite float [-1, 1]
# --------------------------------------------------------------------------- #
def test_to_numpy_accepts_native():
    out = inflect_backend._to_numpy(np.full(100, 0.5, dtype=np.float32), 24_000)
    assert out.dtype == np.float32 and out.ndim == 1 and out.size == 100


@pytest.mark.parametrize("audio,rate", [
    (np.full((2, 100), 0.1, dtype=np.float32), 24_000),   # stereo, not mono
    (np.array([], dtype=np.float32), 24_000),              # empty
    (np.full(100, 0.1, dtype=np.float32), 16_000),         # wrong sample rate
    (np.full(100, 1.5, dtype=np.float32), 24_000),         # outside [-1, 1]
])
def test_to_numpy_rejects_non_native(audio, rate):
    with pytest.raises(RuntimeError):
        inflect_backend._to_numpy(audio, rate)


def test_to_numpy_rejects_nonfinite():
    audio = np.full(100, 0.1, dtype=np.float32)
    audio[10] = float("nan")
    with pytest.raises(RuntimeError):
        inflect_backend._to_numpy(audio, 24_000)


# --------------------------------------------------------------------------- #
# Profile pointer validation: schema, local filename, sha format
# --------------------------------------------------------------------------- #
def test_profile_bad_pointer_schema_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": "wrong", "file": "prof.pt", "sha256": "0" * 64}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr))
    with pytest.raises(ValueError, match="schema"):
        engine._load_profile()
    assert loader.load_calls == []


def test_profile_file_traversal_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": PROFILE_POINTER_SCHEMA, "file": "../evil.pt", "sha256": "0" * 64}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr))
    with pytest.raises(ValueError, match="local companion"):
        engine._load_profile()


def test_profile_bad_sha_format_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"schema": PROFILE_POINTER_SCHEMA, "file": "prof.pt", "sha256": "not-a-sha"}))

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr))
    with pytest.raises(ValueError, match="64 hex"):
        engine._load_profile()


def test_profile_provenance_mismatch_rejected(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader(profile=_ProvenanceProfile(mini_weights=FAKE_MINI_WEIGHTS))
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: "1" * 64)
    ptr, _ = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr))
    with pytest.raises(ValueError, match="manifest"):
        engine._load_profile()
    assert engine._profile is None


def test_profile_loads_from_local_companion_directory(monkeypatch, tmp_path):
    loader = FakeTunerProfileLoader()
    _patch_imported_module(monkeypatch, load=loader.load)
    monkeypatch.setattr(inflect_backend, "_model_weights_sha", lambda package: FAKE_MINI_WEIGHTS)
    ptr, pt = _write_profile_files(tmp_path)

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr))
    engine._verify_release = lambda package: None
    engine._load_adapter = lambda package: setattr(engine, "_adapter", FakeAdapter())
    engine.load()

    assert Path(loader.load_calls[0][0]).parent == tmp_path  # resolved beside the pointer
    assert engine._profile is loader._profile


# --------------------------------------------------------------------------- #
# Release bind + load lifecycle
# --------------------------------------------------------------------------- #
def test_release_manifest_digest_verified_before_import(monkeypatch, tmp_path):
    ran = {"verify": False}

    def fake_verify(package):
        ran["verify"] = True

    _patch_imported_module(monkeypatch, verify=fake_verify, load=lambda *a, **k: FakeAdapter())
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({"schema": "x", "files": {}}))  # digest will not match

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_package=tmp_path))
    with pytest.raises(ValueError, match="pinned digest"):
        engine.load()
    assert ran["verify"] is False  # the package verifier never executed


def test_load_builds_delivery_once(monkeypatch, tmp_path):
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

    engine = InflectMiniEngine(Settings(backend="inflect", inflect_profile=ptr))
    engine._verify_release = lambda package: None
    engine._load_adapter = lambda package: setattr(engine, "_adapter", FakeAdapter())
    engine.load()

    engine.synthesize("hello world")
    engine.synthesize("again now")
    assert len(engine._profile.calls) == 2  # profile reused, not reloaded per request


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_instructions_rejected():
    engine = _stock_engine()
    with pytest.raises(ValueError, match="instructions"):
        engine.synthesize("hello world", instructions="speak calmly")


def test_speed_rejected_except_one():
    engine = _stock_engine()
    with pytest.raises(ValueError, match="speed"):
        engine.synthesize("hello world", speed=1.5)
    engine.synthesize("hello world", speed=1.0)  # accepted
    engine.synthesize("hello world")  # omitted, accepted


def test_not_loaded_raises():
    engine = InflectMiniEngine(Settings(backend="inflect"))
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.synthesize("hello world")


# --------------------------------------------------------------------------- #
# Server routing / readiness; HTTP voice is ignored like other backends
# --------------------------------------------------------------------------- #
def _server_engine(profile=None, *, voice="cassidy"):
    engine = InflectMiniEngine(
        Settings(
            backend="inflect",
            inflect_voice=voice,
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
    return engine


def test_http_voice_ignored(monkeypatch):
    engine = _server_engine()
    monkeypatch.setattr(server, "settings", Settings(backend="inflect"))
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={"input": "hello world", "voice": ANCHOR_VOICE},
        )
    assert res.status_code == 200
    assert res.headers["X-TinyTalk-Backend"] == "inflect"
    # The request voice is ignored; synthesis uses the configured stock voice.
    assert engine._delivery.calls[-1]["voice"] == "cassidy"


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
