"""Compatibility wrappers for the shared speech-quality evaluator."""

from __future__ import annotations

import urllib.request  # kept so existing tests can patch urllib.request.urlopen here

from .quality import transcribe_chunk
from .quality import word_error_rate as _word_error_rate


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Legacy hypothesis-first WER API."""
    return _word_error_rate(reference, hypothesis).rate
