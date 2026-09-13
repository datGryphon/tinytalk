from __future__ import annotations

import json
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .audio import chunk_confidence, to_wav_bytes


@dataclass(frozen=True)
class EditScore:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    rate: float


@dataclass(frozen=True)
class QualityResult:
    wer: float
    cer: float | None
    substitutions: int | None
    deletions: int | None
    insertions: int | None
    prompt_leak: bool
    source: str
    fallback: bool

    def timing_fields(self) -> dict[str, object]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "prompt_leak": self.prompt_leak,
            "wer_source": self.source,
            "wer_fallback": self.fallback,
        }


def normalize_text(text: str) -> str:
    """Normalize text for speech-intelligibility comparison."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in normalized)
    return " ".join(cleaned.split())


def _edit_score(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> tuple[EditScore, tuple[str, ...]]:
    """Levenshtein score plus the hypothesis units aligned as insertions."""
    n = len(reference)
    m = len(hypothesis)
    distances = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        distances[i][0] = i
    for j in range(1, m + 1):
        distances[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = distances[i - 1][j - 1] + int(
                reference[i - 1] != hypothesis[j - 1]
            )
            deletion = distances[i - 1][j] + 1
            insertion = distances[i][j - 1] + 1
            distances[i][j] = min(substitution, deletion, insertion)

    substitutions = 0
    deletions = 0
    insertions = 0
    inserted_units: list[str] = []
    i, j = n, m

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and distances[i][j] == distances[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and distances[i][j] == distances[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
            continue

        if j > 0 and distances[i][j] == distances[i][j - 1] + 1:
            insertions += 1
            inserted_units.append(hypothesis[j - 1])
            j -= 1
            continue

        deletions += 1
        i -= 1

    reference_units = n
    errors = substitutions + deletions + insertions
    return (
        EditScore(
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            reference_units=reference_units,
            rate=0.0 if reference_units == 0 else errors / reference_units,
        ),
        tuple(reversed(inserted_units)),
    )


def word_error_rate(reference: str, hypothesis: str) -> EditScore:
    score, _ = _edit_score(normalize_text(reference).split(), normalize_text(hypothesis).split())
    return score


def char_error_rate(reference: str, hypothesis: str) -> EditScore:
    reference_chars = list(normalize_text(reference).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis).replace(" ", ""))
    score, _ = _edit_score(reference_chars, hypothesis_chars)
    return score


def evaluate_transcript(
    target_text: str,
    transcript: str,
    *,
    prompt_text: str | None = None,
    scaffold_text: str | None = None,
    source: str = "whisper",
    fallback: bool = False,
) -> QualityResult:
    target_words = normalize_text(target_text).split()
    transcript_words = normalize_text(transcript).split()
    wer_score, inserted_words = _edit_score(target_words, transcript_words)
    cer_score = char_error_rate(target_text, transcript)

    target_word_set = set(target_words)
    prompt_words = set(
        normalize_text(" ".join(part for part in (prompt_text, scaffold_text) if part)).split()
    )
    inserted_prompt_words = [
        word
        for word in inserted_words
        if word not in target_word_set and word in prompt_words
    ]

    # One stray shared word is too noisy to classify as prompt leakage. Two
    # aligned insertions from the conditioning/scaffold text is a useful,
    # intentionally conservative signal.
    prompt_leak = len(inserted_prompt_words) >= 2

    return QualityResult(
        wer=wer_score.rate,
        cer=cer_score.rate,
        substitutions=wer_score.substitutions,
        deletions=wer_score.deletions,
        insertions=wer_score.insertions,
        prompt_leak=prompt_leak,
        source=source,
        fallback=fallback,
    )


def _confidence_quality(
    wav: np.ndarray,
    target_text: str,
    *,
    fallback: bool,
) -> QualityResult:
    return QualityResult(
        wer=1.0 - chunk_confidence(wav, len(target_text)),
        cer=None,
        substitutions=None,
        deletions=None,
        insertions=None,
        prompt_leak=False,
        source="confidence",
        fallback=fallback,
    )


def evaluate_audio(
    wav: np.ndarray,
    sample_rate: int,
    *,
    target_text: str,
    endpoint: str,
    prompt_text: str | None = None,
    scaffold_text: str | None = None,
    transcribe: Callable[[bytes, str, str], str] | None = None,
) -> QualityResult:
    """Evaluate generated audio, failing open to the legacy confidence heuristic."""
    if not endpoint:
        return _confidence_quality(wav, target_text, fallback=False)

    transcribe_fn = transcribe or transcribe_chunk
    try:
        transcript = transcribe_fn(to_wav_bytes(wav, sample_rate), target_text, endpoint)
        return evaluate_transcript(
            target_text,
            transcript,
            prompt_text=prompt_text,
            scaffold_text=scaffold_text,
        )
    except Exception:
        return _confidence_quality(wav, target_text, fallback=True)


def quality_is_acceptable(result: QualityResult, threshold: float) -> bool:
    if result.prompt_leak or result.wer > threshold:
        return False
    return result.cer is None or result.cer <= threshold


def quality_rank(result: QualityResult) -> tuple[bool, bool, float, float]:
    """Stable best-candidate ordering without inventing a weighted score."""
    return (
        result.prompt_leak,
        result.source != "whisper",
        result.wer,
        result.cer if result.cer is not None else result.wer,
    )


def transcribe_chunk(wav_bytes: bytes, chunk_text: str, endpoint: str) -> str:
    """POST a WAV chunk to an OpenAI-compatible transcriptions endpoint."""
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
                    raise ConnectionError(f"{url}: 5xx after 3 attempts") from exc
            else:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt < 3:
                time.sleep(backoff[attempt - 1])
            else:
                raise ConnectionError(f"{url}: failed after 3 attempts") from None

    raise RuntimeError("unreachable")
