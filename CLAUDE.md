# CLAUDE.md

This file provides guidance to coding agents working in this repository.

## Project

TinyTalk is an OpenAI-compatible TTS server for modest machines. It owns the
HTTP API, chunking, stitching, audio encoding, service lifecycle, and deployment
surface. Synthesis is provided by one backend per service instance:

- `neutts`: reference-voice NeuTTS/GGUF path;
- `tinytauk`: instruction-controlled AuK-Flash through the separately maintained
  TinyTAuK runtime;
- `omnivoice`: multilingual auto voice, constrained attribute-based voice design,
  and optional reference-audio cloning through k2-fsa/OmniVoice.

The main API is `POST /v1/audio/speech`. The repository exports a NixOS module
as `nixosModules.default`.

## Setup

All current development shells target Python 3.13. Backend extras remain
separate while Torch/Transformers compatibility is qualified.

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

OmniVoice environment:

```bash
nix develop .#omnivoice
pytest tests/test_quality.py tests/test_backend_api.py tests/test_config.py tests/test_omnivoice_backend.py
```

Do not assume separate virtual environments are a permanent architecture
requirement. Keep backend code and optional dependencies isolated, but prefer a
common Python/Torch/Transformers stack when real validation proves it works.

## Conventions

Keep the project simple and direct. Do not add speculative abstractions or
feature flags unrelated to real backend differences.

- Runtime dependencies shared by every backend stay in `[project].dependencies`.
  Backend-specific ML stacks belong in the `neutts`, `tinytauk`, and `omnivoice`
  extras.
- Backend implementations remain separate modules even when their dependency
  versions converge. Environment alignment and code ownership are different
  concerns.
- Imports are selected lazily by `engine.create_engine()` so a service only
  requires its selected backend extra.
- A running TinyTalk service owns exactly one synthesis backend.
- `server.py` owns request validation and the process-level async inference lock.
  Backend engines expose `load()`, `synthesize()`, `loaded`, and `model_name`.
- Long input chunking stays in `chunking.py`; backend implementations should not
  invent competing chunkers.
- Shared correctness evaluation stays in `quality.py`. Backend-specific retry
  policy stays with each backend because their controls and failure modes differ.
- Audio stays float `[-1, 1]` through the pipeline until final WAV/codec encoding.
- NeuTTS tuning behavior stays in `backends/neutts.py`: repeat-penalty rerolls,
  shared quality evaluation, loudness normalization, and F0 boundary smoothing.
  NeuTTS-only audio transforms live in `backends/neutts_audio.py`.
- TinyTAuK-specific model/runtime logic stays in the TinyTAuK repository. The
  TinyTalk adapter should only compose AuK instructions, estimate target duration,
  call TinyTAuK, apply its local quality/retry policy, and perform minimal
  stitching-safe post-processing.
- Do not apply NeuTTS loudness or F0 normalization to TinyTAuK or OmniVoice output.
- TinyTAuK `instructions` are composed into AuK's canonical Instruct-TTS format.
  `speed` adjusts TinyTalk's target-duration estimate.
- TinyTAuK uses a disposable generation during `load()` so its lazy VAE compile
  finishes before `/health` reports ready.
- OmniVoice maps request `instructions` directly to upstream `instruct` and
  request `speed` directly to upstream `speed`.
- OmniVoice `instructions` are comma-separated upstream attribute tags, not
  freeform prose. Keep examples within the fixed gender, age, pitch, whisper,
  accent, and Chinese-dialect vocabularies. Unsupported or mutually exclusive
  values should remain visible as upstream validation errors.
- OmniVoice mode selection is local policy: request instructions select voice
  design; otherwise a configured cached clone prompt selects cloning; otherwise
  the backend uses auto voice.
- OmniVoice configured cloning requires both reference audio and a transcript
  file. Build the reusable clone prompt once during `load()`.
- OmniVoice first-pass generation keeps upstream greedy class sampling. Quality
  retries increase `class_temperature` locally to introduce controlled variation.
- Do not silently change tuned NeuTTS values, TinyTAuK duration behavior, or
  OmniVoice retry policy without an audio/quality check.
- Don't commit unless asked.

## Commands

- NeuTTS unit tests: `nix develop -c pytest`
- TinyTAuK unit tests: `nix develop .#tinytauk -c pytest`
- OmniVoice focused tests: `nix develop .#omnivoice -c pytest tests/test_quality.py tests/test_backend_api.py tests/test_config.py tests/test_omnivoice_backend.py`
- NeuTTS real integration: `TINYTALK_RUN_INTEGRATION=1 pytest tests/integration/test_real_speech.py`
- TinyTAuK real integration: `TINYTALK_RUN_TINYTAUK_INTEGRATION=1 pytest tests/integration/test_tinytauk_real_speech.py`
- OmniVoice real integration: `TINYTALK_RUN_OMNIVOICE_INTEGRATION=1 pytest tests/integration/test_omnivoice_real_speech.py`
- Start server: `uvicorn tinytalk.server:app`
- Build/evaluate flake: `nix flake check` and `nix eval .#nixosModules.default`

Set `TINYTALK_WER_ENDPOINT` during real integration runs when the shared live
WER/CER transcription path also needs validation.

## Architecture

```text
tinytalk/
  server.py                 FastAPI API, lifespan, request lock
  engine.py                 shared SynthesisResult/protocol and backend factory
  quality.py                shared transcription, WER/CER, edit and leak scoring
  wer.py                    compatibility wrapper around shared quality helpers
  backends/
    neutts.py               NeuTTS load/generate/reroll policy
    neutts_audio.py         NeuTTS-only RMS/F0 audio transforms
    tinytauk.py             TinyTAuK adapter and AuK instruction/duration policy
    omnivoice.py            OmniVoice load/mode/clone/retry policy
  config.py                 environment-backed Settings
  chunking.py               sentence/phrase/word chunking
  audio.py                  shared WAV/codec and minimal splice-safe helpers

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

The NeuTTS backend keeps its current behavior: reference codes/text, sampling
overrides, per-chunk repeat-penalty retries, shared quality scoring,
trim/RMS/peak/edge cleanup, F0 boundary smoothing, and inter-chunk silence.

### TinyTAuK backend

TinyTAuK is a library/CLI, not a server. TinyTalk is the serving/orchestration
layer. Keep model architecture, quantization, checkpoint loading, Qwen
conditioning, and VAE compilation code in TinyTAuK.

### OmniVoice backend

The adapter loads one long-lived upstream `OmniVoice` model. CPU loads use
float32; non-CPU devices use float16 unless qualification shows a device-specific
requirement. A configured reference audio/transcript pair is encoded once into a
reusable voice-clone prompt. Request instructions override that configured clone
for the request and select upstream voice-design mode. Instructions must use
upstream's fixed comma-separated attribute vocabulary; they are not semantic
natural-language prompts. With neither instructions nor a configured clone,
generation uses upstream auto voice. Shared WER/CER evaluates every candidate,
while retries raise OmniVoice `class_temperature` from the greedy first-pass
default.

The qualified CPU stack is Python 3.13.13, OmniVoice 0.2.1, Torch 2.8.0+cpu,
and Transformers 5.17.0. One manual sweep measured approximately 2.6-2.8 GiB
peak process RSS for auto/design and approximately 5.1 GiB for cloned generation.
Treat those figures as host-specific capacity guidance rather than guarantees.

Keep pronunciation markup, LoRA, batch inference, request-level reference-audio
uploads, and accelerator-specific optimization out of the generic TinyTalk API
until separately justified.

## Env Var Map

Shared:

| Env var | NixOS option | Default |
| --- | --- | --- |
| `TINYTALK_BACKEND` | `services.tinytalk.backend` | `neutts` |
| `TINYTALK_HOST` | `services.tinytalk.host` | `0.0.0.0` |
| `TINYTALK_PORT` | `services.tinytalk.port` | `9002` |
| `TINYTALK_MAX_CHARS_PER_CHUNK` | `services.tinytalk.maxCharsPerChunk` | `180` |
| `TINYTALK_INTER_CHUNK_SILENCE_MS` | `services.tinytalk.interChunkSilenceMs` | `60` |
| `TINYTALK_MAX_RETRIES` | `services.tinytalk.maxRetries` | `2` |
| `TINYTALK_WER_ENDPOINT` | `services.tinytalk.werEndpoint` | empty |
| `TINYTALK_WER_THRESHOLD` | `services.tinytalk.werThreshold` | `0.25` |

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
| `TINYTALK_WATERMARK` | `services.tinytalk.watermark` |

TinyTAuK:

| Env var | NixOS option | Default |
| --- | --- | --- |
| `TINYTALK_TINYTAUK_MODEL` | `services.tinytalk.tinytaukModel` | `tencent/AuK-Flash` |
| `TINYTALK_TINYTAUK_QWEN_MODEL` | `services.tinytalk.tinytaukQwenModel` | `Qwen/Qwen2.5-Omni-3B` |
| `TINYTALK_TINYTAUK_CHARS_PER_SECOND` | `services.tinytalk.tinytaukCharsPerSecond` | `14.0` |

OmniVoice:

| Env var | NixOS option | Default |
| --- | --- | --- |
| `TINYTALK_OMNIVOICE_MODEL` | `services.tinytalk.omnivoiceModel` | `k2-fsa/OmniVoice` |
| `TINYTALK_OMNIVOICE_DEVICE` | `services.tinytalk.omnivoiceDevice` | `cpu` |
| `TINYTALK_OMNIVOICE_LANGUAGE` | `services.tinytalk.omnivoiceLanguage` | empty |
| `TINYTALK_OMNIVOICE_REF_AUDIO` | `services.tinytalk.omnivoiceRefAudio` | unset |
| `TINYTALK_OMNIVOICE_REF_TEXT` | `services.tinytalk.omnivoiceRefText` | unset |

## Deploy

The NixOS module bootstraps the selected backend's Python packages under
`/var/lib/tinytalk/python` using Python 3.13. The requirements marker includes
the backend, so switching backend forces a clean runtime reinstall. Backend
package lists remain separate until a common Torch/Transformers stack is fully
qualified.

If a host overrides `runtimePackages`, it owns the complete selected backend
runtime set. Keep TinyTAuK pinned to a released tag rather than `main` in
production deployment configuration.
