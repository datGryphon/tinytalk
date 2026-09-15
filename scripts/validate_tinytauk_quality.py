from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from tinytalk.audio import to_wav_bytes
from tinytalk.backends.tinytauk import TinyTAuKEngine
from tinytalk.config import Settings

TARGET = (
    "The deployment completed successfully, although it took considerably longer than expected."
)
RICH_STYLE = (
    "A calm adult technical narrator with clear diction, a natural conversational tone, "
    "moderate pacing, steady volume, and neutral prosody."
)


def _assert_live_quality(chunk: dict, *, threshold: float) -> None:
    assert chunk["wer_source"] == "whisper", chunk
    assert chunk["wer_fallback"] is False, chunk
    assert chunk["cer"] is not None, chunk
    assert chunk["prompt_leak"] is False, chunk
    assert chunk["wer"] <= threshold, chunk
    assert chunk["cer"] <= threshold, chunk
    assert chunk["status"] == "accepted", chunk


def _run_case(
    engine: TinyTAuKEngine,
    *,
    name: str,
    instructions: str | None,
    threshold: float,
    force_all_retries: bool = False,
) -> dict:
    settings = replace(
        engine.settings,
        wer_threshold=-1.0 if force_all_retries else threshold,
        max_retries=2,
    )
    case_engine = TinyTAuKEngine(settings)
    case_engine.tts = engine.tts
    case_engine.sample_rate = engine.sample_rate
    case_engine.loaded = True

    result = case_engine.synthesize(TARGET, instructions=instructions, speed=1.0)
    assert result.timing is not None
    assert len(result.chunks) == 1
    chunk = result.timing.chunks[0]
    details = chunk["attempts_detail"]

    for detail in details:
        assert detail["wer_source"] == "whisper", detail
        assert detail["wer_fallback"] is False, detail
        assert detail["cer"] is not None, detail

    if force_all_retries:
        assert len(details) == 3, details
        seeds = [detail["seed"] for detail in details]
        assert len(set(seeds)) == 3, details
        selected = [detail for detail in details if detail["status"] is not None]
        assert len(selected) == 1, details
        assert selected[0]["status"] == "fallback", details
    else:
        _assert_live_quality(chunk, threshold=threshold)

    out_dir = Path(os.environ.get("TINYTALK_QUALITY_ARTIFACTS", "test_artifacts/tinytauk-quality"))
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{name}.wav"
    wav_path.write_bytes(to_wav_bytes(result.audio, result.sample_rate))

    record = {
        "case": name,
        "instructions": instructions,
        "chunks": len(result.chunks),
        "attempts": len(details),
        "status": chunk["status"],
        "selected_seed": chunk["seed"],
        "selected_gen_seconds": chunk["gen_seconds"],
        "wer": chunk["wer"],
        "cer": chunk["cer"],
        "substitutions": chunk["substitutions"],
        "deletions": chunk["deletions"],
        "insertions": chunk["insertions"],
        "prompt_leak": chunk["prompt_leak"],
        "wer_source": chunk["wer_source"],
        "wer_fallback": chunk["wer_fallback"],
        "attempts_detail": details,
        "wav": str(wav_path),
    }
    print(json.dumps(record, indent=2))
    return record


def main() -> None:
    endpoint = os.environ.get("TINYTALK_WER_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise SystemExit("TINYTALK_WER_ENDPOINT must point at a live transcription service")

    threshold = float(os.environ.get("TINYTALK_WER_THRESHOLD", "0.25"))
    settings = Settings(
        backend="tinytauk",
        max_chars_per_chunk=180,
        max_retries=2,
        wer_endpoint=endpoint,
        wer_threshold=threshold,
    )

    print("Loading TinyTAuK once...")
    engine = TinyTAuKEngine(settings)
    engine.load()

    records = [
        _run_case(
            engine,
            name="canonical-default",
            instructions=None,
            threshold=threshold,
        ),
        _run_case(
            engine,
            name="canonical-imperative-style",
            instructions="Speak in a tired but relieved technical narrator voice.",
            threshold=threshold,
        ),
        _run_case(
            engine,
            name="canonical-rich-style",
            instructions=RICH_STYLE,
            threshold=threshold,
        ),
        _run_case(
            engine,
            name="canonical-forced-reroll",
            instructions=None,
            threshold=threshold,
            force_all_retries=True,
        ),
    ]

    out_dir = Path(os.environ.get("TINYTALK_QUALITY_ARTIFACTS", "test_artifacts/tinytauk-quality"))
    summary = out_dir / "results.json"
    summary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    print("\nPASS: canonical prompting + live WER/CER + retry selection validated")
    print(f"Results: {summary}")


if __name__ == "__main__":
    main()
