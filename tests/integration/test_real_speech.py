import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tinytalk import server
from tinytalk.chunking import split_text
from tinytalk.config import Settings
from tinytalk.engine import TinyTalkEngine

pytestmark = pytest.mark.skipif(
    os.getenv("TINYTALK_RUN_INTEGRATION") != "1",
    reason="set TINYTALK_RUN_INTEGRATION=1 to run real NeuTTS integration tests",
)


CORPUS = Path(__file__).parent.parent / "corpus"
CASES = {p.stem: " ".join(p.read_text().split()) for p in sorted(CORPUS.glob("*.txt"))}


def test_sentencizer_edge_cases_chunk_cleanly():
    for name in ("abbreviations", "initials_and_quotes"):
        chunks = split_text(CASES[name], 180)
        assert len(chunks) >= 1
        assert all(len(chunk) <= 180 for chunk in chunks)

    assert split_text(CASES["abbreviations"], 180) == [CASES["abbreviations"]]


def test_real_speech_outputs_wavs(monkeypatch):
    voices = Path(__file__).parent.parent / "voices"
    settings = Settings(ref_codes=voices / "jo.pt", ref_text=voices / "jo.txt")
    monkeypatch.setattr(server, "engine", TinyTalkEngine(settings))

    artifact_dir = Path("test_artifacts")
    artifact_dir.mkdir(exist_ok=True)

    with TestClient(server.app) as client:
        for name, text in CASES.items():
            response = client.post(
                "/v1/audio/speech",
                json={"input": text, "response_format": "wav"},
                timeout=600,
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "audio/wav"
            assert int(response.headers["X-TinyTalk-Chunks"]) >= 1
            out = artifact_dir / f"{name}.wav"
            out.write_bytes(response.content)
            assert out.stat().st_size > 44
