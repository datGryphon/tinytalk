from __future__ import annotations

import asyncio
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
    async with infer_lock:
        result = await run_in_threadpool(engine.synthesize, payload.input)

    wav = to_wav_bytes(result.audio, result.sample_rate)
    body, media_type = await run_in_threadpool(
        encode_audio, wav, payload.response_format
    )

    return Response(
        content=body,
        media_type=media_type,
        headers={
            "X-TinyTalk-Chunks": str(len(result.chunks)),
            "X-TinyTalk-Chunk-Chars": ",".join(
                str(len(chunk)) for chunk in result.chunks
            ),
            "X-TinyTalk-Model": settings.model,
            "X-TinyTalk-Format": payload.response_format,
        },
    )
