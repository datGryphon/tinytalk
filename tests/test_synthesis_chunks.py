import numpy as np
import pytest

from tinytalk.config import Settings
from tinytalk.engine import SpeechEngine, SynthesisResult
from tinytalk.quality import QualityResult


class StubEngine(SpeechEngine):
    def load(self) -> None:
        self.loaded = True

    def synthesize(
        self,
        text: str,
        *,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        raise NotImplementedError


def test_base_is_abstract():
    with pytest.raises(TypeError):
        SpeechEngine(Settings(), "test")


def test_base_declares_and_initializes_shared_state():
    settings = Settings()
    engine = StubEngine(settings, "test")
    assert vars(engine) == {
        "settings": settings,
        "model_name": "test",
        "sample_rate": 24_000,
        "loaded": False,
    }
    assert SpeechEngine.__annotations__.keys() >= vars(engine).keys()


def test_shared_quality_timing_fields():
    quality = QualityResult(0.1, None, None, None, None, False, "confidence", False)
    assert SpeechEngine._quality_timing_fields(quality) == quality.timing_fields()
    assert SpeechEngine._quality_timing_fields(None) == {
        "wer": None,
        "cer": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "prompt_leak": False,
        "wer_source": None,
        "wer_fallback": False,
    }


@pytest.mark.parametrize("silence_ms", [0, 2])
def test_assembles_chunks_and_aggregates_timings(silence_ms):
    calls = []
    chunks = ["one", "two", "three"]

    def synthesize_chunk(text, index, num_chunks):
        calls.append((text, index, num_chunks))
        return np.full(2, (index + 1) / 4, dtype=np.float32), {"index": index}, index

    engine = StubEngine(Settings(inter_chunk_silence_ms=silence_ms), "test")
    engine.sample_rate = 1000
    result = engine._synthesize_chunks(chunks, synthesize_chunk)

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
    engine = StubEngine(Settings(), "test")
    result = engine._synthesize_chunks(["one"], lambda *args: (audio, {}, 0))
    assert result.audio is audio


def test_generation_failure_stops_before_next_chunk():
    calls = []

    def generate(text, index, count):
        calls.append(text)
        raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        StubEngine(Settings(), "test")._synthesize_chunks(["first", "second"], generate)
    assert calls == ["first"]
