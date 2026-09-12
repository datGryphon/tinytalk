from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .audio import encode_audio, to_wav_bytes
from .config import load_settings
from .engine import TinyTalkEngine

settings = load_settings()
engine = TinyTalkEngine(settings)
infer_lock = asyncio.Lock()
log = logging.getLogger("tinytalk.server")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: str = Field(min_length=1)
    model: str | None = None
    voice: str | None = None
    response_format: Literal["wav", "mp3", "opus"] = "wav"
    speed: float | None = None
    stream: Literal[False] = False

    @model_validator(mode="after")
    def normalize_input(self) -> SpeechRequest:
        self.input = self.input.strip()
        if not self.input:
            raise ValueError("input must not be empty")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(engine.load)
    yield


app = FastAPI(title="tinytalk", version=__version__, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    _exc: RequestValidationError,
) -> Response:
    return Response(status_code=400, content="bad request\n", media_type="text/plain")


@app.get("/health")
async def health() -> Response:
    if not engine.loaded:
        return Response(
            status_code=503, content="not loaded\n", media_type="text/plain"
        )
    return Response(status_code=200, content="ok\n", media_type="text/plain")


@app.post("/v1/audio/speech")
async def create_speech(payload: SpeechRequest) -> Response:
    request_start = time.perf_counter()
    result = None
    error: Exception | None = None
    try:
        async with infer_lock:
            result = await run_in_threadpool(engine.synthesize, payload.input)
        wav = to_wav_bytes(result.audio, result.sample_rate)
        body, media_type = await run_in_threadpool(
            encode_audio, wav, payload.response_format
        )
    except Exception as exc:
        error = exc

    # Full handler wall time: time waiting on infer_lock, synthesis, WAV
    # construction, and encoding — measured whether or not the request failed.
    total_seconds = time.perf_counter() - request_start

    if error is not None:
        # Failure observability: one structured record even on failure. Report
        # only what is safely available (synthesis yields a chunk count); make
        # everything else null rather than fabricating it. The error indicator
        # is the exception type name only — never its message, which may carry
        # user text, and never the input itself.
        log.info(
            json.dumps(
                {
                    "elapsed": total_seconds,
                    "error": type(error).__name__,
                    "chunks": len(result.chunks) if result is not None else None,
                    "audio_seconds": None,
                    "rtf": None,
                    "attempts": None,
                    "wer_fallbacks": None,
                    "timing": None,
                }
            )
        )
        raise error

    timing = result.timing
    chunk_count = len(result.chunks)
    # Total output audio seconds is the actual emitted audio length, which
    # includes inter-chunk silence inserted between chunks. Per-chunk
    # durations do not, so they must not be summed here.
    audio_seconds = len(result.audio) / result.sample_rate
    if timing is not None:
        total_attempts = sum(chunk["attempts"] for chunk in timing.chunks)
        wer_fallbacks = timing.wer_fallbacks
    else:
        total_attempts = 0
        wer_fallbacks = 0
    rtf = total_seconds / audio_seconds if audio_seconds > 0 else 0.0

    # One structured line per request for journald / slice-1 analysis.
    log.info(
        json.dumps(
            {
                "elapsed": total_seconds,
                "audio_seconds": audio_seconds,
                "rtf": rtf,
                "chunks": chunk_count,
                "attempts": total_attempts,
                "wer_fallbacks": wer_fallbacks,
                "timing": timing.chunks if timing is not None else None,
            }
        )
    )

    headers = {
        "X-TinyTalk-Chunks": str(chunk_count),
        "X-TinyTalk-Chunk-Chars": ",".join(
            str(len(chunk)) for chunk in result.chunks
        ),
        "X-TinyTalk-Model": settings.model,
        "X-TinyTalk-Format": payload.response_format,
        "X-TinyTalk-Timing": (
            f"total={total_seconds:.2f};audio={audio_seconds:.2f};rtf={rtf:.2f}"
            f";chunks={chunk_count};attempts={total_attempts};wer_fallbacks={wer_fallbacks}"
        ),
    }

    return Response(content=body, media_type=media_type, headers=headers)
