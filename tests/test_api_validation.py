import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.engine import RequestTiming, SynthesisResult


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
                "text": "chunk one here",
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
                "text": "chunk two here now",
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
