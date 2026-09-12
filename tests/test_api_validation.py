import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.config import Settings
from tinytalk.engine import RequestTiming, TinyTalkEngine, SynthesisResult


class FakeEngine:
    loaded = True
    last_text = None

    def synthesize(self, text):
        self.last_text = text
        return SynthesisResult(
            audio=np.zeros(2400, dtype=np.float32),
            sample_rate=24_000,
            chunks=[text],
        )

    def load(self):
        self.loaded = True


def test_rejects_unknown_response_format(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "response_format": "flac"},
        )
    assert res.status_code == 400


def test_rejects_stream(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "hello", "stream": True})
    assert res.status_code == 400


def test_rejects_blank_input(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "   "})
    assert res.status_code == 400


def test_strips_input_before_synthesis(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "  hello  "})
    assert res.status_code == 200
    assert engine.last_text == "hello"


def test_accepts_ignored_openai_fields_and_returns_wav(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    with TestClient(server.app) as client:
        res = client.post(
            "/v1/audio/speech",
            json={
                "input": "hello",
                "model": "ignored",
                "voice": "ignored",
                "speed": 1.5,
                "unknown": "ignored",
            },
        )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.headers["X-TinyTalk-Chunks"] == "1"


def _make_timing() -> RequestTiming:
    """A two-chunk timing payload with one accepted reroll and a fallback.

    Chunks and attempts are plain dicts (matching what synthesize emits) so the
    server path exercises the real JSON-serializable structure.
    """
    return RequestTiming(
        chunks=[
            {
                "index": 0,
                "attempts": 1,
                "duration": 1.0,
                "repeat_penalty": 1.0,
                "wer": 0.1,
                "wer_source": "whisper",
                "wer_fallback": False,
                "t_f0": 0.05,
                "attempts_detail": [
                    {
                        "attempt": 0,
                        "t_infer": 0.50,
                        "t_dsp": 0.02,
                        "t_wer_check": 0.01,
                        "wer": 0.1,
                        "wer_source": "whisper",
                        "wer_fallback": False,
                        "repeat_penalty": 1.0,
                        "accepted": True,
                    },
                ],
            },
            {
                "index": 1,
                "attempts": 2,
                "duration": 1.5,
                "repeat_penalty": 1.1,
                "wer": 0.2,
                "wer_source": "confidence",
                "wer_fallback": True,
                "t_f0": 0.06,
                "attempts_detail": [
                    {
                        "attempt": 0,
                        "t_infer": 0.60,
                        "t_dsp": 0.03,
                        "t_wer_check": 0.02,
                        "wer": 0.5,
                        "wer_source": "whisper",
                        "wer_fallback": False,
                        "repeat_penalty": 1.0,
                        "accepted": False,
                    },
                    {
                        "attempt": 1,
                        "t_infer": 0.70,
                        "t_dsp": 0.03,
                        "t_wer_check": 0.02,
                        "wer": 0.2,
                        "wer_source": "confidence",
                        "wer_fallback": True,
                        "repeat_penalty": 1.1,
                        "accepted": True,
                    },
                ],
            },
        ],
        wer_fallbacks=1,
    )


class FakeEngineTiming:
    loaded = True
    last_audio: np.ndarray | None = None
    last_sample_rate: int | None = None

    def synthesize(self, text):
        audio = np.zeros(2400, dtype=np.float32)
        self.last_audio = audio
        self.last_sample_rate = 24_000
        return SynthesisResult(
            audio=audio,
            sample_rate=24_000,
            chunks=["chunk one here", "chunk two here now"],
            timing=_make_timing(),
        )

    def load(self):
        self.loaded = True


def test_timing_header_present_and_parseable(monkeypatch):
    engine = FakeEngineTiming()
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        res = client.post("/v1/audio/speech", json={"input": "hello world"})
    assert res.status_code == 200
    assert "X-TinyTalk-Timing" in res.headers
    fields = dict(
        item.split("=") for item in res.headers["X-TinyTalk-Timing"].split(";")
    )
    assert fields["chunks"] == "2"
    assert fields["attempts"] == "3"  # 1 + 2 across the two chunks
    assert fields["wer_fallbacks"] == "1"
    # audio_seconds is the actual emitted audio length, not the sum of the
    # per-chunk durations (which deliberately differ here: 1.0 + 1.5 = 2.5).
    expected_audio = len(engine.last_audio) / engine.last_sample_rate
    assert float(fields["audio"]) == pytest.approx(expected_audio)
    assert expected_audio != 2.5
    assert float(fields["total"]) >= 0.0
    assert float(fields["rtf"]) >= 0.0


def test_timing_logged_as_json_line(monkeypatch, caplog):
    monkeypatch.setattr(server, "engine", FakeEngineTiming())
    with caplog.at_level(logging.INFO, logger="tinytalk.server"):
        caplog.clear()
        with TestClient(server.app) as client:
            res = client.post("/v1/audio/speech", json={"input": "hello world"})
    assert res.status_code == 200

    payloads = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.levelno >= logging.INFO and "timing" in record.getMessage()
    ]
    assert payloads, "expected one JSON timing log line per request"
    payload = payloads[-1]
    assert payload["chunks"] == 2
    assert payload["attempts"] == 3
    assert payload["wer_fallbacks"] == 1
    assert payload["timing"] is not None
    # The server emits the raw chunk list under "timing".
    assert len(payload["timing"]) == 2
    detail = payload["timing"][1]["attempts_detail"]
    assert detail[0]["accepted"] is False
    assert detail[1]["accepted"] is True
    assert detail[1]["wer_fallback"] is True


def _good_audio() -> np.ndarray:
    """Non-silent audio at normal RMS so chunk_confidence is high (accepts easily)."""
    sr = 24_000
    dur = int(sr * (14 / 14.0))
    return (np.sin(np.linspace(0, 2 * np.pi * 440, dur)) * (0.08 / (1 / (2**0.5)))).astype(
        np.float32
    )


def _real_engine_with_fake_tts() -> TinyTalkEngine:
    """A real engine with a fake NeuTTS backbone so synthesize succeeds offline."""
    voices = Path(__file__).parent / "voices"
    settings = Settings(
        ref_codes=voices / "jo.pt",
        ref_text=voices / "jo.txt",
        wer_endpoint="",
        wer_threshold=1.0,  # accept the first good attempt
        max_retries=0,
    )
    engine = TinyTalkEngine(settings)
    engine.ref_codes = [1, 2, 3]
    engine.ref_text = "test reference"
    engine.sample_rate = 24_000
    engine.tts = MagicMock()
    engine.tts.infer = MagicMock(return_value=_good_audio())
    # Skip model download: the lifespan calls load(), which would hit the network.
    engine.load = lambda: None
    return engine


def _timing_records(caplog):
    """Structured timing JSON records emitted by the tinytalk.server logger."""
    records = []
    for record in caplog.records:
        if record.name != "tinytalk.server":
            continue
        try:
            payload = json.loads(record.getMessage())
        except ValueError:
            continue
        if "elapsed" in payload:
            records.append(payload)
    return records


def test_success_log_does_not_contain_input_text(monkeypatch, caplog):
    """A successful request logs timing but never the user input text."""
    secret = "QUOKKAUNIQUE55"
    monkeypatch.setattr(server, "engine", _real_engine_with_fake_tts())
    with caplog.at_level(logging.INFO, logger="tinytalk.server"):
        caplog.clear()
        with TestClient(server.app) as client:
            res = client.post(
                "/v1/audio/speech", json={"input": f"alpha {secret} beta"}
            )
    assert res.status_code == 200

    records = _timing_records(caplog)
    assert len(records) == 1
    blob = json.dumps(records[0])
    assert secret not in blob


def test_synthesis_failure_logs_error_without_input(monkeypatch, caplog):
    """A synthesis failure logs exactly one record with a safe error indicator,
    null metrics, and no user text — even when the exception message contains it.
    The existing 500 HTTP behavior is preserved."""
    secret = "NGUNUNIQUE33"

    class FailingSynthEngine:
        loaded = True

        def synthesize(self, text):
            raise RuntimeError(f"boom involving {text!r}")

        def load(self):
            self.loaded = True

    monkeypatch.setattr(server, "engine", FailingSynthEngine())
    with caplog.at_level(logging.INFO, logger="tinytalk.server"):
        caplog.clear()
        with TestClient(server.app, raise_server_exceptions=False) as client:
            res = client.post(
                "/v1/audio/speech", json={"input": f"hello {secret} world"}
            )
    assert res.status_code == 500

    records = _timing_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record["error"] == "RuntimeError"
    assert record["elapsed"] >= 0.0
    # Synthesis produced no result, so nothing else is reported (null, not 0).
    assert record["chunks"] is None
    assert record["audio_seconds"] is None
    assert record["timing"] is None
    blob = json.dumps(record)
    assert secret not in blob


def test_encoding_failure_logs_error_without_input(monkeypatch, caplog):
    """An encoding failure after successful synthesis still logs one record with
    the safely available chunk count, null timing metrics, and no user text."""
    secret = "PLPKUMUNIQUE21"

    class OkSynthEngine:
        loaded = True

        def synthesize(self, text):
            return SynthesisResult(
                audio=np.zeros(2400, dtype=np.float32),
                sample_rate=24_000,
                chunks=[text],
                timing=None,
            )

        def load(self):
            self.loaded = True

    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg missing")

    engine = OkSynthEngine()
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "encode_audio", boom)
    with caplog.at_level(logging.INFO, logger="tinytalk.server"):
        caplog.clear()
        with TestClient(server.app, raise_server_exceptions=False) as client:
            res = client.post(
                "/v1/audio/speech", json={"input": f"hello {secret} world"}
            )
    assert res.status_code == 500

    records = _timing_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record["error"] == "RuntimeError"
    assert record["elapsed"] >= 0.0
    # Synthesis succeeded, so the chunk count is reported; timing metrics are null.
    assert record["chunks"] == 1
    assert record["audio_seconds"] is None
    assert record["timing"] is None
    blob = json.dumps(record)
    assert secret not in blob
