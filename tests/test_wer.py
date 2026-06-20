"""Tests for WER helpers (unchanged by the reroll refactor)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tinytalk.wer import transcribe_chunk, word_error_rate


class TestWordErrorRate:

    def test_perfect(self):
        assert word_error_rate("hello world", "hello world") == pytest.approx(0.0)

    def test_substitution(self):
        wer = word_error_rate("the big dog", "the big cat")
        assert wer == pytest.approx(1 / 3)

    def test_insertion(self):
        wer = word_error_rate("hello extra world", "hello world")
        assert wer == pytest.approx(1 / 2)

    def test_deletion(self):
        wer = word_error_rate("hello", "hello world")
        assert wer == pytest.approx(1 / 2)

    def test_empty_hypothesis(self):
        assert word_error_rate("", "hello world") == pytest.approx(1.0)

    def test_empty_reference(self):
        assert word_error_rate("hello", "") == pytest.approx(0.0)

    def test_uppercase_punctuation_ignored(self):
        assert word_error_rate("Hello, world!", "hello world") == pytest.approx(0.0)

    def test_multiple_errors(self):
        wer = word_error_rate("a b c d e", "x b z d f")
        assert wer == pytest.approx(3 / 5)


class TestTranscribeChunk:

    def test_returns_text_from_json(self):
        fake_response = json.dumps({"text": "transcribed words"})
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tinytalk.wer.urllib.request.urlopen", return_value=mock_resp):
            result = transcribe_chunk(b"fake-wav", "hello", "http://localhost:9002")

        assert result == "transcribed words"

    def test_raises_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.side_effect = Exception("HTTP 500")

        with patch("tinytalk.wer.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(Exception):
                transcribe_chunk(b"wav", "hi", "http://x")
