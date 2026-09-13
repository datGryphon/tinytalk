from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from tinytalk import engine as engine_module
from tinytalk.audio import to_wav_bytes
from tinytalk.backends.tinytauk import TinyTAuKEngine
from tinytalk.config import Settings

TARGET = (
    "The deployment completed successfully, although it took considerably longer than expected."
)
LOW_STYLE = (
    "A very tired, subdued speaker with low energy, a soft low-pitched voice, slow deliberate "
    "pacing, restrained pitch movement, and long pauses."
)
HIGH_STYLE = (
    "An excited, energetic announcer with a bright high-pitched voice, fast lively pacing, "
    "strong pitch variation, emphatic stress, and animated delivery."
)


def _attach(base: TinyTAuKEngine, settings: Settings) -> TinyTAuKEngine:
    engine = TinyTAuKEngine(settings)
    engine.tts = base.tts
    engine.sample_rate = base.sample_rate
    engine.loaded = True
    return engine


def _passes(detail: dict, threshold: float) -> bool:
    return (
        detail["wer_source"] == "whisper"
        and detail["wer_fallback"] is False
        and detail["cer"] is not None
        and detail["prompt_leak"] is False
        and detail["wer"] <= threshold
        and detail["cer"] <= threshold
    )


def _selected_detail(chunk: dict) -> dict:
    for detail in chunk["attempts_detail"]:
        if detail["seed"] == chunk["seed"] and math.isclose(
            detail["gen_seconds"], chunk["gen_seconds"], rel_tol=0.0, abs_tol=1e-9
        ):
            return detail
    raise AssertionError(f"selected attempt not found: {chunk}")


def _acoustic_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    wav = np.asarray(audio, dtype=np.float64).squeeze()
    rms = float(np.sqrt(np.mean(wav**2)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))

    frame = max(int(sample_rate * 0.040), 1)
    hop = max(int(sample_rate * 0.020), 1)
    frame_rms: list[float] = []
    f0: list[float] = []
    min_lag = max(int(sample_rate / 350.0), 1)
    max_lag = min(int(sample_rate / 70.0), frame - 2)
    window = np.hanning(frame)

    for start in range(0, max(len(wav) - frame + 1, 0), hop):
        chunk = wav[start : start + frame]
        chunk_rms = float(np.sqrt(np.mean(chunk**2)))
        frame_rms.append(chunk_rms)

    active_threshold = max((max(frame_rms) * 0.12) if frame_rms else 0.0, 0.0015)

    for start in range(0, max(len(wav) - frame + 1, 0), hop):
        chunk = wav[start : start + frame]
        chunk_rms = float(np.sqrt(np.mean(chunk**2)))
        if chunk_rms < active_threshold:
            continue
        centered = (chunk - np.mean(chunk)) * window
        corr = np.correlate(centered, centered, mode="full")[frame - 1 :]
        if corr[0] <= 1e-12 or max_lag <= min_lag:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag : max_lag + 1]))
        confidence = float(corr[lag] / corr[0])
        if confidence >= 0.30:
            f0.append(sample_rate / lag)

    active_ratio = (
        float(np.mean(np.asarray(frame_rms) >= active_threshold)) if frame_rms else 0.0
    )
    f0_array = np.asarray(f0, dtype=np.float64)
    if f0_array.size:
        q25, q75 = np.percentile(f0_array, [25, 75])
        f0_median = float(np.median(f0_array))
        f0_iqr = float(q75 - q25)
        f0_std = float(np.std(f0_array))
    else:
        f0_median = None
        f0_iqr = None
        f0_std = None

    return {
        "rms_dbfs": rms_dbfs,
        "active_ratio": active_ratio,
        "f0_median_hz": f0_median,
        "f0_iqr_hz": f0_iqr,
        "f0_std_hz": f0_std,
    }


def _run(
    base: TinyTAuKEngine,
    *,
    settings: Settings,
    name: str,
    instructions: str | None,
    out_dir: Path,
) -> dict:
    engine = _attach(base, settings)
    result = engine.synthesize(TARGET, instructions=instructions, speed=1.0)
    assert result.timing is not None
    assert len(result.chunks) == 1
    chunk = result.timing.chunks[0]
    for detail in chunk["attempts_detail"]:
        assert detail["wer_source"] == "whisper", detail
        assert detail["wer_fallback"] is False, detail
        assert detail["cer"] is not None, detail

    wav_path = out_dir / f"{name}.wav"
    wav_path.write_bytes(to_wav_bytes(result.audio, result.sample_rate))
    selected = _selected_detail(chunk)
    return {
        "case": name,
        "instructions": instructions,
        "chars_per_second": settings.tinytauk_chars_per_second,
        "attempts": len(chunk["attempts_detail"]),
        "selected_attempt": selected["attempt"],
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
        "attempts_detail": chunk["attempts_detail"],
        "acoustics": _acoustic_metrics(result.audio, result.sample_rate),
        "wav": str(wav_path),
    }


def _style_sweep(base: TinyTAuKEngine, settings: Settings, out_dir: Path) -> list[dict]:
    records = [
        _run(
            base,
            settings=replace(settings, max_retries=0),
            name="style-low-energy",
            instructions=LOW_STYLE,
            out_dir=out_dir,
        ),
        _run(
            base,
            settings=replace(settings, max_retries=0),
            name="style-high-energy",
            instructions=HIGH_STYLE,
            out_dir=out_dir,
        ),
    ]
    for record in records:
        selected = record["attempts_detail"][0]
        assert _passes(selected, settings.wer_threshold), record

    low, high = records
    low_f0 = low["acoustics"]["f0_median_hz"]
    high_f0 = high["acoustics"]["f0_median_hz"]
    semitones = None
    if low_f0 and high_f0:
        semitones = 12.0 * math.log2(high_f0 / low_f0)

    print("\nSTYLE CONTRAST")
    print(
        json.dumps(
            {
                "low": low["acoustics"],
                "high": high["acoustics"],
                "high_minus_low_rms_db": (
                    high["acoustics"]["rms_dbfs"] - low["acoustics"]["rms_dbfs"]
                ),
                "high_minus_low_active_ratio": (
                    high["acoustics"]["active_ratio"] - low["acoustics"]["active_ratio"]
                ),
                "high_minus_low_pitch_semitones": semitones,
            },
            indent=2,
        )
    )
    return records


def _correction_fault_injection(
    base: TinyTAuKEngine,
    settings: Settings,
    out_dir: Path,
) -> list[dict]:
    """Exercise the real correction loop without depending on a stochastic model failure.

    Every attempt still reaches the live transcription endpoint. For the first
    attempt only, the harness replaces the returned transcript with a truncated
    prefix after the live ASR call. That deterministically creates a deletion-heavy
    quality failure, so TinyTalk must grow the generation horizon, change seed,
    rerun synthesis, rescore through live ASR, and select a passing later attempt.
    """

    real_transcribe = engine_module.transcribe_chunk
    live_transcripts: list[str] = []
    calls = 0

    def injected_transcribe(wav_bytes: bytes, chunk_text: str, endpoint: str) -> str:
        nonlocal calls
        transcript = real_transcribe(wav_bytes, chunk_text, endpoint)
        live_transcripts.append(transcript)
        calls += 1
        if calls == 1:
            return "The deployment completed successfully."
        return transcript

    engine_module.transcribe_chunk = injected_transcribe
    try:
        record = _run(
            base,
            settings=replace(settings, max_retries=2),
            name="correction-fault-injected",
            instructions=None,
            out_dir=out_dir,
        )
    finally:
        engine_module.transcribe_chunk = real_transcribe

    details = record["attempts_detail"]
    assert len(details) >= 2, record
    first = details[0]
    selected = details[record["selected_attempt"]]

    assert not _passes(first, settings.wer_threshold), record
    assert first["deletions"] is not None and first["insertions"] is not None, record
    assert first["deletions"] > first["insertions"], record
    assert details[1]["seed"] != first["seed"], record
    assert details[1]["gen_seconds"] > first["gen_seconds"], record
    assert record["selected_attempt"] > 0, record
    assert _passes(selected, settings.wer_threshold), record
    assert len(live_transcripts) >= 2, record

    record["live_transcripts"] = live_transcripts
    print("\nCORRECTION FAULT INJECTION")
    print(
        json.dumps(
            {
                "attempts": record["attempts"],
                "selected_attempt": record["selected_attempt"],
                "selected_seed": record["selected_seed"],
                "selected_gen_seconds": record["selected_gen_seconds"],
                "wer": record["wer"],
                "cer": record["cer"],
                "live_transcripts": live_transcripts,
                "attempts_detail": details,
            },
            indent=2,
        )
    )
    return [record]


def main() -> None:
    endpoint = os.environ.get("TINYTALK_WER_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise SystemExit("TINYTALK_WER_ENDPOINT must point at a live transcription service")

    threshold = float(os.environ.get("TINYTALK_WER_THRESHOLD", "0.25"))
    out_dir = Path(
        os.environ.get("TINYTALK_RELEASE_SWEEP_ARTIFACTS", "test_artifacts/tinytauk-release-sweep")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        backend="tinytauk",
        max_chars_per_chunk=180,
        max_retries=2,
        wer_endpoint=endpoint,
        wer_threshold=threshold,
    )

    print("Loading TinyTAuK once...")
    base = TinyTAuKEngine(settings)
    base.load()

    style = _style_sweep(base, settings, out_dir)
    correction = _correction_fault_injection(base, settings, out_dir)
    records = {"style": style, "correction": correction}

    summary = out_dir / "results.json"
    summary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(
        "\nPASS: contrastive style outputs are content-correct and deterministic correction control flow passed"
    )
    print(f"Results: {summary}")
    print("Listen to style-low-energy.wav and style-high-energy.wav for the final style-adherence check.")


if __name__ == "__main__":
    main()
