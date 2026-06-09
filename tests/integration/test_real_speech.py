import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tinytalk.chunking import split_text

pytestmark = pytest.mark.skipif(
    os.getenv("TINYTALK_RUN_INTEGRATION") != "1",
    reason="set TINYTALK_RUN_INTEGRATION=1 to run real NeuTTS integration tests",
)


CASES = {
    "short": "Okay.",
    "multi_sentence": "This is the first sentence. This is the second sentence. This is the third sentence.",
    "abbreviations": (
        "Dr. Ada reviewed fig. 2 with Mr. Smith at 3:00 p.m. and confirmed the service was ready."
    ),
    "initials_and_quotes": (
        'J. R. asked, "Is the service ready?" The answer was yes. '
        "The team shipped the update after one more smoke test."
    ),
    "long_commas": (
        "This sentence is intentionally long, with several comma-separated clauses, "
        "so that the chunker can split at softer phrase boundaries, while still "
        "preserving enough text for a natural sounding utterance."
    ),
    "long_no_punctuation": (
        "today we need the server to handle a long spoken update without punctuation "
        "so it should wrap on normal spaces preserve the voice across each generated "
        "piece avoid dropping the final thought and still sound like one coherent reply "
        "when a caller sends a rough transcript or a quick dictated note"
    ),
    "digest": (
        "Here is a concise digest of the current task. The system should split long input "
        "server-side, synthesize each chunk with the same reference voice, and insert short "
        "pauses between chunks. This test is meant to expose truncation, unnatural transitions, "
        "and pathological long-sentence behavior before another service depends on the server."
    ),
}


def test_sentencizer_edge_cases_chunk_cleanly():
    for name in ("abbreviations", "initials_and_quotes"):
        chunks = split_text(CASES[name], 180)
        assert len(chunks) >= 1
        assert all(len(chunk) <= 180 for chunk in chunks)

    assert split_text(CASES["abbreviations"], 180) == [CASES["abbreviations"]]


def test_real_speech_outputs_wavs():
    from tinytalk.server import app

    artifact_dir = Path("test_artifacts")
    artifact_dir.mkdir(exist_ok=True)

    with TestClient(app) as client:
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
