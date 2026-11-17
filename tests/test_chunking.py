import pytest

from tinytalk.chunking import split_tagged_text, split_text


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("Okay.", 300, ["Okay."]),
        (
            "One sentence. Two sentence! Three?",
            25,
            ["One sentence.", "Two sentence! Three?"],
        ),
        (
            "Dr. Ada checked fig. 2. Then she replied.",
            300,
            ["Dr. Ada checked fig. 2. Then she replied."],
        ),
        (
            "one two three four five six seven eight nine ten",
            15,
            ["one two three", "four five six", "seven eight", "nine ten"],
        ),
        ("abcdefghij", 4, ["abcd", "efgh", "ij"]),
        ("   \n\t  ", 300, []),
    ],
)
def test_split_text_cases(text, max_chars, expected):
    assert split_text(text, max_chars) == expected


def test_long_sentence_splits_on_commas_before_whitespace():
    text = "alpha beta gamma, delta epsilon zeta, eta theta iota."
    chunks = split_text(text, 24)
    assert all(len(chunk) <= 24 for chunk in chunks)
    assert chunks[0].endswith(",")


def test_tagged_split_preserves_distinct_native_tags():
    chunks = split_tagged_text("First. [laugh] Second. [clear_throat] Last.", 14)
    text = " ".join(chunks)
    assert text.count("[laugh]") == text.count("[clear_throat]") == 1
    assert text.index("[laugh]") < text.index("[clear_throat]")
    assert all(len(chunk) <= 14 for chunk in chunks)


def test_tagged_split_rejects_undersized_limit():
    with pytest.raises(ValueError, match="fit every native tag"):
        split_tagged_text("Hello [laugh]", 3)


def test_tagged_split_keeps_plain_text_behavior():
    text = "One sentence. Another sentence."
    assert split_tagged_text(text, 15) == split_text(text, 15)
