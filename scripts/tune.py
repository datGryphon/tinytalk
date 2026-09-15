#!/usr/bin/env python3
"""Reproducible NeuTTS WER parameter-sweep harness.

Synthesizes tests/corpus/*.txt probes, scores with tinytalk.wer, writes structured JSON.

Usage:
  # single eval (no sweep flags = loaded settings only)
  python scripts/tune.py

  # sweep temperature + repeat_penalty
  python scripts/tune.py --temperature 0.6,0.7,0.8 --repeat-penalty 1.0,1.15,1.3
"""

from __future__ import annotations

import argparse
import itertools
import json
import textwrap
from dataclasses import replace
from pathlib import Path

from tinytalk.audio import to_wav_bytes
from tinytalk.backends.neutts import NeuTTSEngine
from tinytalk.config import load_settings
from tinytalk.wer import transcribe_chunk, word_error_rate

CORPUS = Path(__file__).parent.parent / "tests" / "corpus"
VOICES = Path(__file__).parent.parent / "tests" / "voices"

PROBES = {p.stem: " ".join(p.read_text().split()) for p in sorted(CORPUS.glob("*.txt"))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WER parameter-sweep harness for the TinyTalk NeuTTS backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
          Examples:
            python scripts/tune.py
            python scripts/tune.py --temperature 0.7,0.8 --repeat-penalty 1.0,1.15
            python scripts/tune.py --temperature 0.7,0.8 --repeat-penalty 1.0,1.15 --max-chars-per-chunk 150,180
        """),
    )
    parser.add_argument("--ref-codes", type=Path, default=VOICES / "jo.pt")
    parser.add_argument("--ref-text", type=Path, default=VOICES / "jo.txt")
    parser.add_argument("--temperature", type=str, help="comma-separated temperatures")
    parser.add_argument("--repeat-penalty", type=str, help="comma-separated repeat penalties")
    parser.add_argument("--max-chars-per-chunk", type=str, help="comma-separated max chars per chunk")
    parser.add_argument("--inter-chunk-silence-ms", type=str, help="comma-separated silence ms")
    parser.add_argument("--max-retries", type=str, help="comma-separated max retries")
    parser.add_argument("--wer-threshold", type=str, help="comma-separated WER thresholds")
    parser.add_argument("--wer-endpoint", default="http://localhost:9003", help="WER transcription endpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("test_artifacts"))
    parser.add_argument("--out-json", type=Path, help="write JSON results to this path")
    args = parser.parse_args()

    sweep_fields: list[tuple[str, str, type]] = [
        ("temperature", args.temperature, float),
        ("repeat_penalty", args.repeat_penalty, float),
        ("max_chars_per_chunk", args.max_chars_per_chunk, int),
        ("inter_chunk_silence_ms", args.inter_chunk_silence_ms, int),
        ("max_retries", args.max_retries, int),
        ("wer_threshold", args.wer_threshold, float),
    ]
    field_values: dict[str, list] = {}
    for field_name, raw, typ in sweep_fields:
        if raw is not None:
            field_values[field_name] = [typ(v) for v in raw.split(",")]

    overrides = {
        "backend": "neutts",
        "ref_codes": args.ref_codes,
        "ref_text": args.ref_text,
        "wer_endpoint": args.wer_endpoint,
    }
    base_settings = replace(
        load_settings(),
        **{k: v for k, v in overrides.items() if v is not None},
    )

    for field_name, raw, typ in sweep_fields:
        if field_name not in field_values:
            field_values[field_name] = [getattr(base_settings, field_name)]

    product_args = [field_values[fn] for fn, _, _ in sweep_fields]
    combos = list(itertools.product(*product_args))

    engine = NeuTTSEngine(base_settings)
    engine.load()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    best_result: dict | None = None
    best_wer = 1.0

    for combo in combos:
        combo_dict = dict(zip([fn for fn, _, _ in sweep_fields], combo))
        engine.settings = replace(engine.settings, **combo_dict)

        human_parts = [f"{k}={v}" for k, v in combo_dict.items()]
        print(f"{' | '.join(human_parts)}")

        combo_dir = args.out_dir / "_".join(f"{k}{v}" for k, v in combo_dict.items())
        combo_dir.mkdir(parents=True, exist_ok=True)

        per_probe_wer: dict[str, float] = {}
        wers: list[float] = []

        for name, text in PROBES.items():
            result = engine.synthesize(text)
            wav_bytes = to_wav_bytes(result.audio, result.sample_rate)
            (combo_dir / f"{name}.wav").write_bytes(wav_bytes)
            transcript = transcribe_chunk(wav_bytes, text, args.wer_endpoint)
            wer = word_error_rate(transcript, text)
            per_probe_wer[name] = round(wer, 4)
            wers.append(wer)

        mean_wer = sum(wers) / len(wers)
        result_entry = {
            "params": {k: v for k, v in combo_dict.items()},
            "audio_dir": str(combo_dir),
            "per_probe_wer": per_probe_wer,
            "mean_wer": round(mean_wer, 4),
        }
        results.append(result_entry)
        print(f"  mean WER: {mean_wer:.1%}")

        if mean_wer < best_wer:
            best_wer = mean_wer
            best_result = result_entry

    if best_result is not None:
        params = best_result["params"]
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\nbest: {param_str} mean_wer={best_result['mean_wer']}")

    if args.out_json is not None:
        results.sort(key=lambda r: r["mean_wer"])
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON written to {args.out_json}")


if __name__ == "__main__":
    main()
