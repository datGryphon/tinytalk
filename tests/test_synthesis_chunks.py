import numpy as np
import pytest

from tinytalk.engine import synthesize_chunks


@pytest.mark.parametrize("silence_ms", [0, 2])
def test_assembles_chunks_and_aggregates_timings(silence_ms):
    calls = []
    chunks = ["one", "two", "three"]

    def synthesize_chunk(text, index, num_chunks):
        calls.append((text, index, num_chunks))
        return np.full(2, (index + 1) / 4, dtype=np.float32), {"index": index}, index

    result = synthesize_chunks(chunks, 1000, silence_ms, synthesize_chunk)

    assert calls == [("one", 0, 3), ("two", 1, 3), ("three", 2, 3)]
    gap = [0.0] * silence_ms
    np.testing.assert_array_equal(result.audio, [0.25, 0.25, *gap, 0.5, 0.5, *gap, 0.75, 0.75])
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 1000
    assert result.chunks == chunks
    assert result.timing.chunks == [{"index": 0}, {"index": 1}, {"index": 2}]
    assert result.timing.wer_fallbacks == 3


def test_single_chunk_keeps_native_audio():
    audio = np.array([0.1, -0.2], dtype=np.float32)
    result = synthesize_chunks(["one"], 24000, 60, lambda *args: (audio, {}, 0))
    assert result.audio is audio
