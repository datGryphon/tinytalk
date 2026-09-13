# CLAUDE.md

This file provides guidance to coding agents working in this repository.

## Project

TinyTalk is an OpenAI-compatible TTS server for modest machines. It owns the
HTTP API, chunking, stitching, audio encoding, service lifecycle, and deployment
surface. Synthesis is provided by one backend per service instance:

- `neutts`: reference-voice NeuTTS/GGUF path;
- `tinytauk`: instruction-controlled AuK-Flash through the separately maintained
  TinyTAuK runtime.

The main API is `POST /v1/audio/speech`. The repository exports a NixOS module
as `nixosModules.default`.

## Setup

NeuTTS/default environment:

```bash
nix develop
pytest
```

TinyTAuK environment:

```bash
nix develop .#tinytauk
pytest
```

The shells intentionally use separate virtual environments. TinyTAuK v0.1.0
requires Python 3.13 / Torch 2.7.1 while current NeuTTS uses a newer
Torch/Transformers stack. Do not try to solve this by co-installing both ML
backends into one environment.

## Conventions

Keep the project simple and direct. Do not add speculative abstractions or
feature flags unrelated to real backend differences.

- Runtime dependencies shared by every backend stay in `[project].dependencies`.
  Backend-specific ML stacks belong in the `neutts` and `tinytauk` extras.
- Imports stay at module scope except in `engine.create_engine()`. That factory
  intentionally imports only the selected backend because the backend dependency
  stacks are mutually incompatible.
- A running TinyTalk service owns exactly one synthesis backend. If two backends
  are needed simultaneously, run two service instances rather than combining
  their Python environments.
- `server.py` owns request validation and the process-level async inference lock.
  Backend engines expose `load()`, `synthesize()`, `loaded`, and `model_name`.
- Long input chunking stays in `chunking.py`; backend implementations should not
  invent competing chunkers.
- Audio stays float `[-1, 1]` through the pipeline until final WAV/codec encoding.
- NeuTTS tuning behavior stays in `backends/neutts.py`: repeat-penalty rerolls,
  WER/confidence selection, loudness normalization, and F0 boundary smoothing.
- TinyTAuK-specific model/runtime logic stays in the TinyTAuK repository. The
  TinyTalk adapter should only compose AuK instructions, estimate target duration,
  call TinyTAuK, and perform minimal stitching-safe post-processing.
- Do not apply NeuTTS loudness or F0 normalization to TinyTAuK output. AuK uses
  natural-language instructions for expressive pitch/loudness/prosody, and those
  transforms would erase intended behavior.
- TinyTAuK `instructions` are composed into AuK's instruction format. `speed`
  adjusts TinyTalk's target-duration estimate. `voice` remains ignored until
  TinyTAuK implements reference-audio generation.
- TinyTAuK uses a disposable generation during `load()` so its lazy VAE compile
  finishes before `/health` reports ready.
- Do not silently change tuned NeuTTS values or the TinyTAuK duration baseline.
  Changes require an audio/quality check.
- Don't commit unless asked.

## Commands

- NeuTTS unit tests: `nix develop -c pytest`
- TinyTAuK unit tests: `nix develop .#tinytauk -c pytest`
- NeuTTS real integration: `TINYTALK_RUN_INTEGRATION=1 pytest tests/integration/test_real_speech.py`
- TinyTAuK real integration: `TINYTALK_RUN_TINYTAUK_INTEGRATION=1 pytest tests/integration/test_tinytauk_real_speech.py`
- Start server: `uvicorn tinytalk.server:app`
- Build/evaluate flake: `nix flake check` and `nix eval .#nixosModules.default`

## Architecture

```text
tinytalk/
  server.py                 FastAPI API, lifespan, request lock
  engine.py                 shared SynthesisResult/protocol and backend factory
  backends/
    neutts.py               NeuTTS load/generate/reroll/postprocess path
    tinytauk.py             TinyTAuK adapter and AuK instruction/duration policy
  config.py                 environment-backed Settings
  chunking.py               sentence/phrase/word chunking
  audio.py                  WAV/codec helpers and NeuTTS audio transforms
  wer.py                    optional remote WER scoring helper

nix/
  module.nix                backend-aware NixOS service/runtime selection
  tinytalk-prestart.sh      Python target bootstrap

tests/
  test_*.py                 shared and backend-specific unit tests
  integration/              opt-in real synthesis tests
  corpus/                   probe text
  voices/                   NeuTTS reference voices for tests
```

Request flow:

```text
POST /v1/audio/speech
  -> server validation
  -> one async inference lock
  -> selected backend synthesize()
  -> shared chunk concatenation result
  -> WAV bytes
  -> optional ffmpeg MP3/Opus encoding
  -> response
```

### NeuTTS backend

The existing NeuTTS backend keeps its current behavior: reference codes/text,
sampling overrides, per-chunk retry/WER selection, trim/RMS/peak/edge cleanup,
F0 boundary smoothing, and inter-chunk silence.

NeuTTS limitations belong upstream unless TinyTalk orchestration can address them
without forking NeuTTS.

### TinyTAuK backend

TinyTAuK is a library/CLI, not a server. TinyTalk is the serving/orchestration
layer. The adapter:

1. loads one long-lived `TinyTAuK.from_pretrained()` instance;
2. performs one disposable warmup generation;
3. chunks text with TinyTalk's existing chunker;
4. estimates generation duration from characters/second and request `speed`;
5. converts optional request `instructions` into AuK's natural-language style
   description;
6. uses deterministic-but-distinct seeds across chunks;
7. trims splice-edge silence, peak-limits, edge-fades, and concatenates.

Do not move model architecture, quantization, checkpoint loading, Qwen
conditioning, or VAE compilation code into TinyTalk. Those belong in TinyTAuK.

## Env Var Map

Shared:

| Env var | NixOS option | Default |
| --- | --- | --- |
| `TINYTALK_BACKEND` | `services.tinytalk.backend` | `neutts` |
| `TINYTALK_HOST` | `services.tinytalk.host` | `0.0.0.0` |
| `TINYTALK_PORT` | `services.tinytalk.port` | `9002` |
| `TINYTALK_MAX_CHARS_PER_CHUNK` | `services.tinytalk.maxCharsPerChunk` | `180` |
| `TINYTALK_INTER_CHUNK_SILENCE_MS` | `services.tinytalk.interChunkSilenceMs` | `60` |

NeuTTS:

| Env var | NixOS option |
| --- | --- |
| `TINYTALK_MODEL` | `services.tinytalk.model` |
| `TINYTALK_CODEC` | `services.tinytalk.codec` |
| `TINYTALK_BACKBONE_DEVICE` | `services.tinytalk.backboneDevice` |
| `TINYTALK_REF_CODES` | `services.tinytalk.refCodes` |
| `TINYTALK_REF_TEXT` | `services.tinytalk.refText` |
| `TINYTALK_TEMPERATURE` | `services.tinytalk.temperature` |
| `TINYTALK_REPEAT_PENALTY` | `services.tinytalk.repeatPenalty` |
| `TINYTALK_REPEAT_PENALTY_REROLL_STEP` | `services.tinytalk.repeatPenaltyRerollStep` |
| `TINYTALK_MAX_RETRIES` | `services.tinytalk.maxRetries` |
| `TINYTALK_WER_ENDPOINT` | `services.tinytalk.werEndpoint` |
| `TINYTALK_WER_THRESHOLD` | `services.tinytalk.werThreshold` |
| `TINYTALK_WATERMARK` | `services.tinytalk.watermark` |

TinyTAuK:

| Env var | NixOS option | Default |
| --- | --- | --- |
| `TINYTALK_TINYTAUK_MODEL` | `services.tinytalk.tinytaukModel` | `tencent/AuK-Flash` |
| `TINYTALK_TINYTAUK_QWEN_MODEL` | `services.tinytalk.tinytaukQwenModel` | `Qwen/Qwen2.5-Omni-3B` |
| `TINYTALK_TINYTAUK_CHARS_PER_SECOND` | `services.tinytalk.tinytaukCharsPerSecond` | `14.0` |

## Deploy

The NixOS module bootstraps backend-specific Python packages under
`/var/lib/tinytalk/python`. NeuTTS uses Python 3.12 by default; TinyTAuK uses
Python 3.13. The package-requirements marker includes the backend so switching
backend forces a clean runtime reinstall.

If a host overrides `runtimePackages`, it owns the complete selected backend
runtime set. Keep TinyTAuK pinned to a released tag rather than `main` in
production deployment configuration.
