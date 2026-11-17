from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from . import __version__
from .audio import encode_audio, to_wav_bytes
from .config import load_settings
from .engine import SynthesisResult, UnknownVoiceError, create_engine
from .timing import RequestLog

settings = load_settings()
engine = create_engine(settings)
infer_lock = asyncio.Lock()
log = logging.getLogger("tinytalk.server")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "mp3", "opus"] = "wav"
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    stream: Literal[False] = False

    @field_validator("instructions")
    @classmethod
    def normalize_instructions(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Uvicorn configures its handlers before application startup. Reuse those
    # handlers for TinyTalk's app-specific logger so INFO telemetry is visible
    # in service/dev logs without changing the logger name used by tests and
    # external log filters.
    uvicorn_log = logging.getLogger("uvicorn.error")
    if uvicorn_log.handlers and not log.handlers:
        for handler in uvicorn_log.handlers:
            log.addHandler(handler)
        log.setLevel(uvicorn_log.level)
        log.propagate = False

    await run_in_threadpool(engine.load)
    yield


app = FastAPI(title="tinytalk", version=__version__, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, _exc: RequestValidationError) -> Response:
    return Response(status_code=400, content="bad request\n", media_type="text/plain")


@app.get("/health")
async def health() -> Response:
    if not engine.loaded:
        return Response(status_code=503, content="not loaded\n", media_type="text/plain")
    return Response(status_code=200, content="ok\n", media_type="text/plain")


@app.post("/v1/audio/speech")
async def create_speech(payload: SpeechRequest) -> Response:
    request_start = time.perf_counter()
    result: SynthesisResult | None = None
    try:
        async with infer_lock:
            result = await run_in_threadpool(
                engine.synthesize,
                payload.input,
                instructions=payload.instructions,
                speed=payload.speed,
                voice=payload.voice,
            )
        wav = to_wav_bytes(result.audio, result.sample_rate)
        body, media_type = await run_in_threadpool(encode_audio, wav, payload.response_format)
    except UnknownVoiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.info(RequestLog(
            elapsed=time.perf_counter() - request_start,
            error=type(exc).__name__,
            backend=settings.backend,
            chunks=len(result.chunks) if result is not None else None,
            audio_seconds=None,
            rtf=None,
            attempts=None,
            wer_fallbacks=None,
            timing=None,
        ).model_dump_json(exclude_unset=True))
        raise

    total_seconds = time.perf_counter() - request_start
    timing = result.timing
    chunk_count = len(result.chunks)
    audio_seconds = len(result.audio) / result.sample_rate
    total_attempts = timing.total_attempts if timing else 0
    wer_fallbacks = timing.wer_fallbacks if timing else 0
    rtf = total_seconds / audio_seconds if audio_seconds > 0 else 0.0

    log.info(RequestLog(
        elapsed=total_seconds,
        audio_seconds=audio_seconds,
        rtf=rtf,
        backend=settings.backend,
        chunks=chunk_count,
        attempts=total_attempts,
        wer_fallbacks=wer_fallbacks,
        timing=timing.chunks if timing else None,
    ).model_dump_json(exclude_unset=True))

    headers = {
        "X-TinyTalk-Chunks": str(chunk_count),
        "X-TinyTalk-Chunk-Chars": ",".join(str(len(chunk)) for chunk in result.chunks),
        "X-TinyTalk-Backend": settings.backend,
        "X-TinyTalk-Model": getattr(engine, "model_name", settings.model),
        "X-TinyTalk-Format": payload.response_format,
        "X-TinyTalk-Timing": (
            f"total={total_seconds:.2f};audio={audio_seconds:.2f};rtf={rtf:.2f}"
            f";chunks={chunk_count};attempts={total_attempts};wer_fallbacks={wer_fallbacks}"
        ),
    }
    return Response(content=body, media_type=media_type, headers=headers)
