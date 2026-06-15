#!/usr/bin/env python3
"""Validate tinytalk TTS output against original text using local whisper.

Usage:
  python scripts/validate.py --text "Hello world" --output out.wav
  python scripts/validate.py --audio test_artifacts/long_no_punctuation.wav --expected "..."
"""

from __future__ import annotations

import argparse
import string
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import whisper

TINYTALK_URL = "http://localhost:9002/v1/audio/speech"
WHISPER_MODEL = "small.en"

_model = whisper.load_model(WHISPER_MODEL)


def _synthesize(text: str, output_path: Path) -> None:
    """Call the local tinytalk server and write WAV bytes to output_path."""
    with httpx.Client(timeout=300) as client:
        resp = client.post(
            TINYTALK_URL,
            json={
                "model": "tinytalk",
                "input": text,
                "voice": "default",
                "response_format": "wav",
                "speed": 1.0,
            },
        )
        resp.raise_for_status()
        output_path.write_bytes(resp.content)


_PUNCT = str.maketrans("", "", string.punctuation)


def _words(text: str) -> list[str]:
    # Drop punctuation and case: low signal for TTS accuracy, inflates WER.
    return text.lower().translate(_PUNCT).split()


def _word_error_rate(original: str, transcribed: str) -> float:
    orig_words = _words(original)
    trans_words = _words(transcribed)
    if not orig_words:
        return 0.0
    matcher = SequenceMatcher(None, orig_words, trans_words)
    edits = sum(1 for op in matcher.get_opcodes() if op[0] in ("replace", "delete"))
    return edits / len(orig_words)


def validate_wav(wav_path: Path, expected_text: str) -> float:
    """Transcribe a WAV file and return its word error rate vs expected text."""
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()

    print(f"  Audio: {sr}Hz, {channels}ch, {width * 8}-bit, {frames / sr:.1f}s")

    transcribed = _model.transcribe(str(wav_path))["text"].strip()
    print(f"  Original: {expected_text[:80]}...")
    print(f"  Whisper:  {transcribed[:80]}...")

    wer = _word_error_rate(expected_text, transcribed)
    print(f"  WER: {wer:.1%}")
    return wer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate tinytalk TTS output against expected text"
    )
    parser.add_argument("--text", help="Original text to synthesize")
    parser.add_argument("--output", type=Path, help="Output WAV file path")
    parser.add_argument("--audio", type=Path, help="Existing WAV file to transcribe")
    parser.add_argument("--expected", help="Expected text for --audio mode")
    args = parser.parse_args()

    if args.audio:
        if not args.expected:
            print("Error: --expected required with --audio", file=sys.stderr)
            sys.exit(1)
        wer = validate_wav(args.audio, args.expected)
    elif args.text:
        output_path = args.output or Path("validate_out.wav")
        print(f"Synthesizing: {args.text[:60]}...")
        _synthesize(args.text, output_path)
        print(f"Saved: {output_path}")
        print("Transcribing...")
        wer = validate_wav(output_path, args.text)
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'PASS' if wer < 0.3 else 'FAIL'}: WER={wer:.1%}")
    sys.exit(0 if wer < 0.3 else 1)


if __name__ == "__main__":
    main()
