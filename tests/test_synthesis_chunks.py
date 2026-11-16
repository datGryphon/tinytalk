import numpy as np
import pytest

from tinytalk.config import Settings
from tinytalk.engine import ChunkResult, SpeechBackend, SpeechEngine, SynthesisRequest
from tinytalk.quality import QualityResult


class StubBackend(SpeechBackend[SynthesisRequest]):
    calls: list[tuple[str, int, int]]
    finalized: list[SynthesisRequest]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "test")
        self.calls = []
        self.finalized = []

    def load(self) -> None:
        self.loaded = True

    def prepare_request(self, *, instructions: str | None, speed: float | None) -> SynthesisRequest:
        return SynthesisRequest(instructions, speed)

    def synthesize_chunk(
        self, request: SynthesisRequest, chunk_text: str, index: int, num_chunks: int,
    ) -> ChunkResult:
        self.calls.append((chunk_text, index, num_chunks))
        return np.full(2, (index + 1) / 4, dtype=np.float32), {"index": index}, index

    def finalize_request(self, request: SynthesisRequest) -> None:
        self.finalized.append(request)


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        SpeechBackend(Settings(), "test")


def test_engine_delegates_resident_state_and_loading():
    settings = Settings()
    backend = StubBackend(settings)
    engine = SpeechEngine(backend)
    assert engine.settings is settings
    assert engine.model_name == "test"
    assert engine.sample_rate == 24_000
    assert not engine.loaded
    engine.load()
    assert backend.loaded and engine.loaded
    assert vars(engine) == {"backend": backend}


def test_shared_quality_timing_fields():
    quality = QualityResult(0.1, None, None, None, None, False, "confidence", False)
    assert SpeechBackend._quality_timing_fields(quality) == quality.timing_fields()
    assert SpeechBackend._quality_timing_fields(None) == {
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
    backend = StubBackend(Settings(max_chars_per_chunk=5, inter_chunk_silence_ms=silence_ms))
    backend.sample_rate = 1000
    engine = SpeechEngine(backend)
    engine.load()
    result = engine.synthesize("one two three", instructions="calm", speed=1.2)

    assert backend.calls == [("one", 0, 3), ("two", 1, 3), ("three", 2, 3)]
    gap = [0.0] * silence_ms
    np.testing.assert_array_equal(result.audio, [0.25, 0.25, *gap, 0.5, 0.5, *gap, 0.75, 0.75])
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 1000
    assert result.chunks == ["one", "two", "three"]
    assert result.timing.chunks == [{"index": 0}, {"index": 1}, {"index": 2}]
    assert result.timing.wer_fallbacks == 3
    assert backend.finalized == [SynthesisRequest("calm", 1.2)]


def test_single_chunk_keeps_native_audio(monkeypatch):
    audio = np.array([0.1, -0.2], dtype=np.float32)
    backend = StubBackend(Settings())
    monkeypatch.setattr(backend, "synthesize_chunk", lambda *args: (audio, {}, 0))
    engine = SpeechEngine(backend)
    engine.load()
    result = engine.synthesize("one")
    assert result.audio is audio


def test_generation_failure_finalizes_and_stops_before_next_chunk(monkeypatch):
    backend = StubBackend(Settings(max_chars_per_chunk=6))

    def generate(request, text, index, count):
        backend.calls.append((text, index, count))
        raise ValueError("generation failed")

    monkeypatch.setattr(backend, "synthesize_chunk", generate)
    engine = SpeechEngine(backend)
    engine.load()
    with pytest.raises(ValueError, match="generation failed"):
        engine.synthesize("first second")
    assert backend.calls == [("first", 0, 2)]
    assert backend.finalized == [SynthesisRequest(None, None)]


def test_requests_have_independent_state():
    backend = StubBackend(Settings())
    engine = SpeechEngine(backend)
    engine.load()
    engine.synthesize("one")
    engine.synthesize("two")
    assert len(backend.finalized) == 2
    assert backend.finalized[0] is not backend.finalized[1]


def test_unloaded_engine_rejects_request():
    engine = SpeechEngine(StubBackend(Settings()))
    with pytest.raises(RuntimeError, match="engine not loaded"):
        engine.synthesize("one")
