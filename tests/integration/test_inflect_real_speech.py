"""Opt-in real Inflect Mini integration tests.

Requires the pinned official package at TINYTALK_INFLECT_PACKAGE and, for the
profile cases, a local tuner profile pointer at TINYTALK_INFLECT_PROFILE. These
run only when explicitly enabled and never touch reference audio or enrollment
encoders. WAVs are preserved outside the repo.
"""
import builtins
import io
import os
import sys
import tempfile
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from tinytalk import server
from tinytalk.backends.inflect import InflectMiniEngine
from tinytalk.config import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("TINYTALK_RUN_INFLECT_INTEGRATION") != "1",
    reason="set TINYTALK_RUN_INFLECT_INTEGRATION=1 to run real Inflect integration tests",
)

_ENCODER_MODULES = ("inno_kokoro", "inno_ecapa", "speechbrain")


def _package() -> Path:
    return Path(os.getenv("TINYTALK_INFLECT_PACKAGE", ""))


def _profile() -> Path | None:
    raw = os.getenv("TINYTALK_INFLECT_PROFILE")
    return Path(raw) if raw else None


def _settings(*, voice: str = "cassidy", profile: Path | None = None) -> Settings:
    return Settings(
        backend="inflect",
        inflect_package=_package(),
        inflect_voice=voice,
        inflect_profile=profile,
        max_chars_per_chunk=180,
        inter_chunk_silence_ms=60,
    )


def _artifact_dir() -> Path:
    override = os.getenv("TINYTALK_INFLECT_ARTIFACTS")
    directory = Path(override) if override else Path(tempfile.mkdtemp(prefix="inflect-artifacts-"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _wav_format(raw: bytes) -> tuple[int, int, int]:
    with io.BytesIO(raw) as handle:
        with wave.open(handle, "rb") as wav:
            return wav.getnchannels(), wav.getsampwidth(), wav.getframerate()


def _imported_encoders() -> list[str]:
    """NeuTTS enrollment encoders must not be imported on the Inflect serving path."""
    return [name for name in sys.modules if any(name == m or name.startswith(m + ".") for m in _ENCODER_MODULES)]


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


def _record(engine):
    """Forward engine.synthesize while capturing each returned SynthesisResult."""
    recorded: list = []
    original = engine.synthesize

    def _wrapper(text, *, instructions=None, speed=None):
        result = original(text, instructions=instructions, speed=speed)
        recorded.append(result)
        return result

    engine.synthesize = _wrapper
    return recorded


def test_inflect_stock_event_http(monkeypatch):
    package = _package()
    if not package:
        pytest.skip("set TINYTALK_INFLECT_PACKAGE to run real Inflect integration tests")

    engine = InflectMiniEngine(_settings(voice="cassidy"))
    monkeypatch.setattr(server, "settings", engine.settings)
    monkeypatch.setattr(server, "engine", engine)
    _deny_reference(monkeypatch)
    recorded = _record(engine)

    plain_text = "The system is operating normally."
    event_text = "That was surprising. [laugh] Let us continue."
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        delivery_before = engine._delivery  # captured immediately after lifespan entry
        responses = [
            client.post(
                "/v1/audio/speech",
                json={"input": plain_text, "voice": "hope"},
                timeout=1800,
            ),
            client.post(
                "/v1/audio/speech",
                json={"input": plain_text, "voice": "hope"},
                timeout=1800,
            ),
            client.post(
                "/v1/audio/speech",
                json={"input": event_text, "voice": "hope"},
                timeout=1800,
            ),
        ]

    assert len(recorded) == 3
    assert recorded[0].timing.chunks[0]["mode"] == "stock"
    assert recorded[1].timing.chunks[0]["mode"] == "stock"
    assert recorded[2].timing.chunks[0]["mode"] == "events"

    assert engine._delivery is delivery_before  # resident identity unchanged across calls
    assert engine.adapter_loads == 1  # one resident adapter across all calls
    assert _imported_encoders() == []

    for index, resp in enumerate(responses):
        assert resp.status_code == 200, (index, resp.text)
        assert resp.headers["X-TinyTalk-Backend"] == "inflect"
        assert resp.headers["X-TinyTalk-Model"] == engine.model_name
        channels, sampwidth, framerate = _wav_format(resp.content)
        assert (channels, sampwidth, framerate) == (1, 2, 24000), (index, channels, sampwidth, framerate)

    (_artifact_dir() / "cassidy_stock.wav").write_bytes(responses[0].content)
    (_artifact_dir() / "cassidy_event.wav").write_bytes(responses[-1].content)


def test_inflect_conditioned_profile_http(monkeypatch):
    package = _package()
    profile = _profile()
    if not package or profile is None:
        pytest.skip("set TINYTALK_INFLECT_PACKAGE and TINYTALK_INFLECT_PROFILE")

    # Configured stock voice is irrelevant while a profile is loaded.
    engine = InflectMiniEngine(_settings(voice="james", profile=profile))
    monkeypatch.setattr(server, "settings", engine.settings)
    monkeypatch.setattr(server, "engine", engine)
    _deny_reference(monkeypatch)
    recorded = _record(engine)

    conditioned_text = "The tuner profile applies on the resident model."
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        delivery_before = engine._delivery
        profile_before = engine._profile
        responses = [
            client.post(
                "/v1/audio/speech",
                json={"input": conditioned_text, "voice": voice},
                timeout=1800,
            )
            for voice in ("hope", "mark", "sophia")
        ]

    for index, resp in enumerate(responses):
        assert resp.status_code == 200, (index, resp.text)
        assert resp.headers["X-TinyTalk-Backend"] == "inflect"
        channels, sampwidth, framerate = _wav_format(resp.content)
        assert (channels, sampwidth, framerate) == (1, 2, 24000)

    assert len(recorded) == 3
    for index, result in enumerate(recorded):
        chunk = result.timing.chunks[0]
        assert chunk["mode"] == "profile", (index, chunk)
        assert chunk["profile_applied"] is True, (index, chunk)
        assert chunk["full_anchor_fallback"] is False, (index, chunk)

    assert engine._delivery is delivery_before  # identity cached, not reloaded
    assert engine._profile is profile_before
    assert engine.adapter_loads == 1
    assert _imported_encoders() == []

    for index, resp in enumerate(responses):
        (_artifact_dir() / f"profile_{index}.wav").write_bytes(resp.content)


def test_inflect_conditioned_event_rejected():
    package = _package()
    profile = _profile()
    if not package or profile is None:
        pytest.skip("set TINYTALK_INFLECT_PACKAGE and TINYTALK_INFLECT_PROFILE")

    engine = InflectMiniEngine(_settings(profile=profile))
    engine.load()

    # A conditioned profile with event markup is rejected by the backend, not by HTTP.
    with pytest.raises(ValueError):
        engine.synthesize("Wait. [yawn] What was that?")
