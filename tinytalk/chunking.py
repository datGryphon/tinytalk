import re

import spacy

_SOFT_SPLIT_RE = re.compile(r"(?<=[,;])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_NLP = spacy.blank("en")
_NLP.add_pipe("sentencizer")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_tagged_text(text: str, max_chars: int) -> list[str]:
    """Keep native sound/delivery tags atomic while using the shared splitter."""
    pattern = r"\[[^\]]*\]|<[^>]*>"
    marker = "\uE000"
    if marker in text:
        raise ValueError("input contains a reserved markup placeholder")
    tags = re.findall(pattern, text)
    if not tags:
        return split_text(text, max_chars)
    longest = max(map(len, tags))
    if max_chars < longest:
        raise ValueError("max_chars_per_chunk must fit every native tag")
    protected = re.sub(pattern, marker, text)
    budget = max_chars
    while True:
        iterator = iter(tags)
        chunks = [
            re.sub(marker, lambda _: next(iterator), chunk)
            for chunk in split_text(protected, budget)
        ]
        overflow = max(map(len, chunks)) - max_chars
        if overflow <= 0:
            return chunks
        budget = max(1, budget - overflow)


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text for NeuTTS without making callers know about context limits.

    Boundary preference:
    1. sentences ending in . ! ?
    2. comma/semicolon phrases
    3. whitespace hard wrap
    4. raw character slice as last resort
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    normalized = normalize_text(text)
    if not normalized:
        return []

    return _pack(_pieces(normalized, max_chars), max_chars)


def _pieces(text: str, max_chars: int):
    for sentence in (sent.text.strip() for sent in _NLP(text).sents):
        for phrase in _SOFT_SPLIT_RE.split(sentence):
            phrase = phrase.strip()
            if len(phrase) <= max_chars:
                yield phrase
            else:
                yield from _word_wrap(phrase, max_chars)


def _word_wrap(text: str, max_chars: int):
    current = ""
    for word in text.split(" "):
        if len(word) > max_chars:
            if current:
                yield current
                current = ""
            yield from (word[i : i + max_chars] for i in range(0, len(word), max_chars))
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            yield current
            current = word

    if current:
        yield current


def _pack(pieces, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    return chunks
