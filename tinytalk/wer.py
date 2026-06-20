"""Tier 2: WER-based chunk quality check via OpenAI-compatible transcriptions API."""

from __future__ import annotations

import io
import json
import string
import time
import urllib.error
import urllib.request

_PUNCT = str.maketrans("", "", string.punctuation)


def _words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split."""
    return text.lower().translate(_PUNCT).split()


def word_error_rate(hypothesis: str, reference: str) -> float:
    """WER = (substitutions + deletions + insertions) / max(len(ref_words), 1).

    Uses Wagner-Fischer DP to compute a proper Levenshtein edit distance.
    """
    ref = _words(reference)
    hyp = _words(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0

    # Wagner-Fischer: one-dimensional DP array
    dp = list(range(m + 1))
    for r in ref:
        prev = dp.copy()
        dp[0] = prev[0] + 1
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return float(dp[m]) / n


def transcribe_chunk(wav_bytes: bytes, chunk_text: str, endpoint: str) -> str:
    """POST a chunk WAV to an OpenAI-compatible /v1/audio/transcriptions endpoint.

    Returns the ``text`` field from the JSON response. Raises on HTTP error or
    timeout. Retries up to 3 attempts on transient failures with 2s/4s backoff.
    Per-request timeout is 60 seconds.
    """
    boundary = "----TinyTalkBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    url = f"{endpoint}/v1/audio/transcriptions"
    backoff = [2, 4]
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["text"]
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600:
                if attempt < 3:
                    time.sleep(backoff[attempt - 1])
                else:
                    raise ConnectionError(
                        f"{url}: 5xx after 3 attempts"
                    ) from exc
            else:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt < 3:
                time.sleep(backoff[attempt - 1])
            else:
                raise ConnectionError(
                    f"{url}: failed after 3 attempts"
                ) from None
