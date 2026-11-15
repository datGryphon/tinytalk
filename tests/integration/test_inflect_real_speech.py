"""Opt-in real Inflect Mini integration tests.

Requires the pinned official package at TINYTALK_INFLECT_PACKAGE and, for the
profile cases, a local tuner profile pointer at TINYTALK_INFLECT_PROFILE. These
run only when explicitly enabled and never touch reference audio or enrollment
encoders. WAVs are preserved outside the repo.
"""
import builtins
import importlib.util
import io
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from tinytalk import server
from tinytalk.backends.inflect import VOICES, InflectMiniEngine
from tinytalk.config import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("TINYTALK_RUN_INFLECT_INTEGRATION") != "1",
    reason="set TINYTALK_RUN_INFLECT_INTEGRATION=1 to run real Inflect integration tests",
)

_ENCODER_MODULES = (
    "inno_kokoro",
    "inno_kokoro.",
    "speechbrain.",
    "inno_ecapa",
    "inno_ecapa.",
)


def _alias() -> str:
    return os.getenv("TINYTALK_INFLECT_PROFILE_VOICE", "bert")


def _settings(profile=None, voice="cassidy") -> Settings:
    return Settings(
        backend="inflect",
        inflect_package=Path(os.getenv("TINYTALK_INFLECT_PACKAGE", "")),
        inflect_voice=voice,
        inflect_profile=(
            profile
            if profile is not None
            else (Path(os.getenv("TINYTALK_INFLECT_PROFILE")) if os.getenv("TINYTALK_INFLECT_PROFILE") else None)
        ),
        inflect_profile_voice=_alias(),
        max_chars_per_chunk=180,
        inter_chunk_silence_ms=60,
    )


def _profile_anchor_voice(engine: InflectMiniEngine):
    sid = engine._profile.as_dict().get("anchor_sid")
    return VOICES[sid] if sid is not None else None


def _residential_encoders_resident() -> list[str]:
    return [name for name in sys.modules if name.startswith(_ENCODER_MODULES)]


def _artifact_dir() -> Path:
    override = os.getenv("TINYTALK_INFLECT_ARTIFACTS")
    directory = Path(override) if override else Path(tempfile.mkdtemp(prefix="inflect-artifacts-"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _wav_format(raw: bytes) -> tuple[int, int, int]:
    with io.BytesIO(raw) as handle:
        with wave.open(handle, "rb") as wav:
            return wav.getnchannels(), wav.getsampwidth(), wav.getframerate()


def _deny_reference(monkeypatch):
    """Fail if the optional private reference path is ever opened."""
    deny = os.getenv("TINYTALK_INFLECT_DENY_REFERENCE")
    if not deny:
        return
    real_open = builtins.open

    def _guarded(path, *args, **kwargs):
        if deny in str(path):
            raise AssertionError(f"reference read blocked: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded)
    monkeypatch.setattr(io, "open", _guarded)


def _assert_enrollment_absent():
    """Enrollment/voice-clone stacks must not be importable in this serving env."""
    for name in ("speechbrain", "torchaudio", "inno_kokoro", "inno_ecapa"):
        assert importlib.util.find_spec(name) is None, f"{name} should be absent"


def test_inflect_stock_profile_event_sequence_with_profile(monkeypatch):
    alias = _alias()
    settings = _settings(profile=os.getenv("TINYTALK_INFLECT_PROFILE"))
    if settings.inflect_profile is None:
        pytest.skip("profile pointer required for the profile calls in this sequence")

    engine = InflectMiniEngine(settings)
    monkeypatch.setattr(server, "settings", settings)
    monkeypatch.setattr(server, "engine", engine)

    # Reference no-read audit: fail if the private reference path is opened.
    _deny_reference(monkeypatch)

    # Record the real SynthesisResult for each HTTP call while still forwarding.
    original = engine.synthesize
    recorded: list = []

    def _wrapper(text, *, instructions=None, speed=None, voice=None):
        result = original(text, instructions=instructions, speed=speed, voice=voice)
        recorded.append((text, voice, result))
        return result

    engine.synthesize = _wrapper

    artifact_dir = _artifact_dir()

    with TestClient(server.app) as client:
        # Lifespan initializes the engine (engine.load()); assert readiness.
        health = client.get("/health")
        assert health.status_code == 200, health.text
        delivery_before = engine._delivery
        profile_before = engine._profile

        r1 = client.post(
            "/v1/audio/speech",
            json={"input": "TinyTalk now speaks through the resident Inflect Mini.", "voice": "cassidy"},
            timeout=1800,
        )
        r2 = client.post(
            "/v1/audio/speech",
            json={"input": "The tuner profile applies on the resident model.", "voice": alias},
            timeout=1800,
        )
        r3 = client.post(
            "/v1/audio/speech",
            json={"input": "Please do not [laugh] during the presentation.", "voice": "cassidy"},
            timeout=1800,
        )
        r4 = client.post(
            "/v1/audio/speech",
            json={"input": "The profile holds on the same resident model.", "voice": alias},
            timeout=1800,
        )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r3.status_code == 200, r3.text
    assert r4.status_code == 200, r4.text

    for index, resp in enumerate((r1, r2, r3, r4)):
        channels, sampwidth, framerate = _wav_format(resp.content)
        assert (channels, sampwidth, framerate) == (1, 2, 24000), (index, channels, sampwidth, framerate)
        assert resp.headers["X-TinyTalk-Backend"] == "inflect"
        assert resp.headers["X-TinyTalk-Model"] == engine.model_name

    assert len(recorded) == 4

    stock_result = recorded[0][2]
    stock_chunk = stock_result.timing.chunks[0]
    assert stock_chunk["mode"] == "stock"
    assert stock_chunk["voice"] == "cassidy"

    profile_result = recorded[1][2]
    profile_chunk = profile_result.timing.chunks[0]
    assert profile_chunk["mode"] == "profile"
    assert profile_chunk["voice"] == alias
    assert profile_chunk["profile_applied"] is True
    assert profile_chunk["full_anchor_fallback"] is False

    events_result = recorded[2][2]
    event_chunk = events_result.timing.chunks[0]
    assert event_chunk["mode"] == "events"
    assert event_chunk["events"]["profiles"][0]["tag"] == "laugh"
    assert event_chunk["events"]["profiles"][0]["voice"] == "cassidy"
    assert "speech_metadata" not in event_chunk["events"]
    assert "text" not in event_chunk["events"]
    for seg in event_chunk["events"]["joins"]["segments"]:
        assert "text" not in seg
        assert "phones" not in seg

    profile_result2 = recorded[3][2]
    profile_chunk2 = profile_result2.timing.chunks[0]
    assert profile_chunk2["mode"] == "profile"
    assert profile_chunk2["profile_applied"] is True
    assert profile_chunk2["full_anchor_fallback"] is False

    # One resident delivery/profile across all four calls; no repeat load.
    assert engine._delivery is delivery_before
    assert engine._profile is profile_before
    assert engine.adapter_loads == 1
    assert _residential_encoders_resident() == []
    _assert_enrollment_absent()

    # Preserve all four WAVs outside the repo.
    names = ("cassidy_stock.wav", "bert_profile.wav", "cassidy_laugh.wav", "bert_profile2.wav")
    bodies = (r1.content, r2.content, r3.content, r4.content)
    for name, body in zip(names, bodies):
        out = artifact_dir / name
        out.write_bytes(body)
        assert out.stat().st_size > 44
        assert not out.resolve().is_relative_to(Path.cwd().resolve())


def test_inflect_conditioned_event_rejected_at_api(monkeypatch):
    alias = _alias()
    settings = _settings(profile=os.getenv("TINYTALK_INFLECT_PROFILE"))
    if settings.inflect_profile is None:
        pytest.skip("profile required to exercise the conditioned+event rejection")
    engine = InflectMiniEngine(settings)
    engine.load()
    monkeypatch.setattr(server, "settings", settings)
    monkeypatch.setattr(server, "engine", engine)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Wait. [yawn] What was that?", "voice": alias},
            timeout=1800,
        )
    assert response.status_code == 400, response.text


def test_inflect_missing_profile_pointer_is_denied():
    settings = _settings(profile=Path("/nonexistent/bert-profile.json"))
    engine = InflectMiniEngine(settings)
    with pytest.raises((FileNotFoundError, OSError)):
        engine.load()


def test_inflect_unknown_voice_returns_400(monkeypatch):
    settings = _settings()
    engine = InflectMiniEngine(settings)
    engine.load()
    monkeypatch.setattr(server, "settings", settings)
    monkeypatch.setattr(server, "engine", engine)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "not-a-voice"},
            timeout=1800,
        )
    assert response.status_code == 400, response.text
