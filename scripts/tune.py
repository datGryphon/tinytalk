#!/usr/bin/env python3
"""Synthesize the probe corpus and report WER, to tune a voice's settings.

Runs every probe through the engine into test_artifacts/ and prints the WER for
each (via local whisper). To tune, set the generation env vars and re-run, e.g.
`TINYTALK_TEMPERATURE=0.8 python scripts/tune.py`, and compare. Uses the bundled
jo voice unless overridden.

Usage:
  python scripts/tune.py
  python scripts/tune.py --ref-codes voice.pt --ref-text voice.txt
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tinytalk.audio import to_wav_bytes
from tinytalk.config import load_settings
from tinytalk.engine import TinyTalkEngine

from validate import validate_wav

ARTIFACTS = Path("test_artifacts")
VOICES = Path(__file__).parent.parent / "tests" / "voices"
CORPUS = Path(__file__).parent.parent / "tests" / "corpus"

PROBES = {p.stem: " ".join(p.read_text().split()) for p in sorted(CORPUS.glob("*.txt"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-codes", type=Path, default=VOICES / "jo.pt")
    parser.add_argument("--ref-text", type=Path, default=VOICES / "jo.txt")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--max-chars-per-chunk", type=int)
    parser.add_argument("--inter-chunk-silence-ms", type=int)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()

    overrides = {
        "ref_codes": args.ref_codes,
        "ref_text": args.ref_text,
        "temperature": args.temperature,
        "repeat_penalty": args.repeat_penalty,
        "max_chars_per_chunk": args.max_chars_per_chunk,
        "inter_chunk_silence_ms": args.inter_chunk_silence_ms,
    }
    settings = replace(
        load_settings(), **{k: v for k, v in overrides.items() if v is not None}
    )
    engine = TinyTalkEngine(settings)
    engine.load()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"temperature={settings.temperature} repeat_penalty={settings.repeat_penalty} "
        f"max_chars={settings.max_chars_per_chunk} silence_ms={settings.inter_chunk_silence_ms}\n"
    )
    wers = []
    for name, text in PROBES.items():
        result = engine.synthesize(text)
        path = args.out_dir / f"{name}.wav"
        path.write_bytes(to_wav_bytes(result.audio, result.sample_rate))
        print(f"[{name}]")
        wers.append(validate_wav(path, text))
        print()
    print(f"mean WER: {sum(wers) / len(wers):.1%}")


if __name__ == "__main__":
    main()
