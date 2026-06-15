# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

tinytalk is an OpenAI-compatible TTS server backed by NeuTTS, designed for low VRAM environments. The main API is `POST /v1/audio/speech` which accepts OpenAI-style JSON and returns synthesized audio. Long inputs are auto-chunked server-side. Exports a NixOS module (`nixosModules.default`).

## Setup

```bash
nix develop   # creates/activates .venv and installs everything on first entry
```

The dev flake (`flake.nix`) provides python, uv, and ffmpeg, sets
`LD_LIBRARY_PATH` for PyPI wheels on NixOS, and bootstraps one `.venv` with all
deps. There are no optional dependency groups beyond `[test]`.

## Conventions

Keep this project simple and direct. It is a personal tool, so don't add
deployment hedging, feature flags, or speculative abstractions.

- **Imports at module scope only.** No in-function imports, no `try/except
  ImportError`. The flake guarantees every dependency is present, so never guard
  or lazy-import.
- **Dependencies stay flat.** Runtime deps go in `[project].dependencies`. Only
  test-only tools (pytest, httpx, openai-whisper) go in `[test]`. No other extras.
- **No on/off knob for behavior that should just be on.** Real tunables
  (temperature, repeat_penalty, chunk size, silence) live in three places that
  must stay in sync: the `Settings` field + `TINYTALK_*` parse in `config.py`,
  and the matching `services.tinytalk.*` option + env in `nix/module.nix`.
- **NeuTTS generation params are hardcoded in the library** with no public API.
  Override them by wrapping the backbone's `create_completion` in
  `engine.load()` (`_apply_generation_settings`). Do not fork NeuTTS.
- **Audio stays float `[-1, 1]` through the pipeline** until the final
  `to_wav_bytes`. Returning int16 mid-pipeline clips to full scale once
  concatenated with float chunks.
- **Validate locally.** `scripts/validate.py` transcribes with in-process
  whisper (`small.en`), no external service. WER strips punctuation and case
  because they are low signal for TTS accuracy.
- **NeuTTS limitations belong upstream.** It voices abbreviations and times
  literally ("3:00 p.m." → "three zero zero pee dot em"), and fully-repetitive
  text (e.g. "This is the Nth sentence" ×3) loops or drops clauses at any
  temperature. Fix these in whatever generates the script, not here.
- **Generation stability is voice-dependent.** When adding a reference voice,
  run `scripts/tune.py` (it synthesizes `tests/corpus/` and reports WER), then
  set `TINYTALK_TEMPERATURE` / `TINYTALK_REPEAT_PENALTY` and re-run to compare.
  Low temperature drops content, high loops it, and the balance differs per voice.
- **Probe texts live in `tests/corpus/` only** (one `.txt` per case). Both the
  integration test and tune.py load them, so never inline test text in either.
- Don't silently change tuned values. Don't commit unless asked.

## Commands

- Run all tests: `pytest`
- Run a single test file: `pytest tests/test_chunking.py`
- Run a single test: `pytest tests/test_chunking.py::test_some_name`
- Run integration tests (downloads the model on first run, uses the bundled
  `jo` voice in `tests/voices/` by default). Synthesis is slow on CPU and much
  faster on a CUDA GPU:
  ```bash
  TINYTALK_RUN_INTEGRATION=1 pytest tests/integration
  ```
  Override the voice with `TINYTALK_REF_CODES` / `TINYTALK_REF_TEXT` if needed.
- Start the server for manual testing: `uvicorn tinytalk.server:app`
- Build the Nix flake: `nix build`
- Evaluate the NixOS module: `nix eval .#nixosModules.default`

## Architecture

```
tinytalk/
  server.py      — FastAPI app: lifespan (engine load), /health, /v1/audio/speech
  engine.py      — TinyTalkEngine: loads NeuTTS, validates refs, synthesize() does chunking, trimming, F0 normalization, silence padding
  config.py      — Settings dataclass loaded entirely from env vars
  chunking.py    — split_text() with sentence/phrase/word boundaries, word-wrap fallback, overflow handling
  audio.py       — encode_audio() (ffmpeg), to_wav_bytes(), trim_edge_silence(), silence(), compute_f0_mean()/normalize_f0() (librosa + parselmouth PSOLA)

tests/
  test_api_validation.py   — bad payloads return 400
  test_audio.py            — encode, wav_bytes, trim_edge_silence, silence helpers
  test_chunking.py         — split_text, normalize_text
  test_health.py           — /health returns 200 when engine loaded
  integration/test_real_speech.py — end-to-end synthesis (TINYTALK_RUN_INTEGRATION=1)
  corpus/                  — probe texts, one per `.txt`. Single source for the integration test and tune.py — add/edit cases here only
  voices/                  — reference voices for tests. `jo` (jo.pt/jo.txt) pulled from neutts upstream samples

nix/
  module.nix               — NixOS service: options, systemd unit, prestart bootstrap
  tinytalk-prestart.sh     — Python bootstrap: installs deps into /var/lib/tinytalk/python,
                           — spec file (index+packages) guards against reinstall

scripts/
  encode_reference.py      — encode reference WAV → ref_codes.pt (dev tool, not needed at serve time)
  validate.py              — validate_wav(): transcribe a wav with local whisper, return WER (reused by tune.py)
  tune.py                  — synthesize the tests/corpus probes to test_artifacts and print WER per probe. Override settings with flags or TINYTALK_* env and re-run to tune
```

Flow: request → `server.py` validates → `engine.synthesize()` calls `chunking.split_text()` → each chunk sent through NeuTTS → edge silence trimmed per-chunk → F0 of later chunks normalized to the first chunk's mean pitch → inter-chunk silence inserted → WAV encoded → optional ffmpeg transcode (MP3/Opus) via `audio.encode_audio()` → Response with `X-TinyTalk-*` metadata headers.

Key constraints:
- Single `infer_lock` (asyncio.Lock) ensures sequential inference because the GGUF backbone isn't thread-safe.
- Long inputs are auto-chunked at sentence/phrase boundaries. `stream: true` is rejected.
- Reference files (`ref_codes.pt`, `ref_text.txt`) must exist before first start. The NixOS module does not create them. Generate with `scripts/encode_reference.py`.
- CPU-only is the default runtime. CUDA requires overriding `runtimeIndexUrl`, `runtimeExtraIndexUrls`, and `runtimePackages` in the NixOS module, plus extending `LD_LIBRARY_PATH` with the NVIDIA driver and CUDA libs.

## Env Var Map

| Env var              | NixOS option                  | Default                 |
|----------------------|-------------------------------|-------------------------|
| `TINYTALK_MODEL`     | `services.tinytalk.model`     | `neuphonic/neutts-nano-q4-gguf` |
| `TINYTALK_CODEC`     | `services.tinytalk.codec`     | `neuphonic/neucodec-onnx-decoder-int8` |
| `TINYTALK_BACKBONE_DEVICE` | `services.tinytalk.backboneDevice` | `cpu`       |
| `TINYTALK_REF_CODES` | `services.tinytalk.refCodes`  | `/var/lib/tinytalk/ref_codes.pt` |
| `TINYTALK_REF_TEXT`  | `services.tinytalk.refText`   | `/var/lib/tinytalk/ref_text.txt` |
| `TINYTALK_HOST`      | `services.tinytalk.host`      | `0.0.0.0`               |
| `TINYTALK_PORT`      | `services.tinytalk.port`      | `9002`                  |
| `TINYTALK_MAX_CHARS_PER_CHUNK` | `services.tinytalk.maxCharsPerChunk` | `180`       |
| `TINYTALK_INTER_CHUNK_SILENCE_MS` | `services.tinytalk.interChunkSilenceMs` | `60` |
| `TINYTALK_TEMPERATURE` | `services.tinytalk.temperature` | `1.0` |
| `TINYTALK_REPEAT_PENALTY` | `services.tinytalk.repeatPenalty` | `1.0` |

## Deploy

This repo exports a NixOS module (`nixosModules.default`) meant to be consumed
as a flake input by a host configuration. If that host overrides
`runtimePackages`, any new runtime dependency must be added there too. For
example, F0 needs `librosa` and `praat-parselmouth`, so a host pinning its own
package list will crash on import without them.
