# tinytalk

OpenAI-compatible TTS server for modest machines. TinyTalk owns the HTTP API,
chunking, stitching, output encoding, shared correctness checks, and service
lifecycle while synthesis is provided by a selectable backend.

Supported backends:

- **NeuTTS** — reference-voice/GGUF synthesis;
- **TinyTAuK** — instruction-controlled AuK-Flash through
  [`datGryphon/tinytauk`](https://github.com/datGryphon/tinytauk);
- **OmniVoice** — multilingual auto voice, constrained attribute-based voice
  design, and optional reference-audio voice cloning through `k2-fsa/OmniVoice`.

The flake exports `nixosModules.default`.

## Backend model licensing

TinyTalk itself is MIT-licensed. Backend packages and model weights keep their
own upstream terms; TinyTalk's license does not grant rights to those models.

| Backend | Default model stack | Upstream licensing notes |
| --- | --- | --- |
| **NeuTTS** | [`neuphonic/neutts-nano-q4-gguf`](https://huggingface.co/neuphonic/neutts-nano-q4-gguf) + [`neuphonic/neucodec-onnx-decoder-int8`](https://huggingface.co/neuphonic/neucodec-onnx-decoder-int8) | The default Nano Q4 model repo is gated and currently reports `license: other`; review its bundled `LICENCE` before deployment. The default INT8 NeuCodec repo is Apache-2.0 and also gated. Other selectable NeuTTS checkpoints can differ; for example, [`neuphonic/neutts-air-q4-gguf`](https://huggingface.co/neuphonic/neutts-air-q4-gguf) currently reports Apache-2.0. |
| **TinyTAuK** | [`tencent/AuK-Flash`](https://huggingface.co/tencent/AuK-Flash) + [`Qwen/Qwen2.5-Omni-3B`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | AuK-Flash is MIT-licensed. The default Qwen conditioner uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE), which grants non-commercial use only unless you obtain a separate commercial license from Alibaba Cloud. |
| **OmniVoice** | [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) | OmniVoice code is Apache-2.0, while the published pretrained weights are CC-BY-NC/non-commercial. |

These notes describe the default model IDs in this repository as of September
2026. If you point TinyTalk at different checkpoints or a custom TinyTAuK
profile, check the licenses for the complete model stack you actually deploy.

## Python environments

`pyproject.toml` and `uv.lock` are the source of truth for TinyTalk's Python
runtime. The project pins the qualified Torch/Torchaudio/Transformers versions,
the PyTorch CPU index, the NeuCodec fork, and backend package versions once.
Development, CI, and the NixOS service all consume that same lock.

From a source checkout, sync only the backend you need:

```bash
uv sync --frozen --extra neutts
uv sync --frozen --extra tinytauk
uv sync --frozen --extra omnivoice
```

For development, add the `test` extra. To validate coexistence of all three
backends, select all three extras in the same sync.

The base `tinytalk` project remains backend-light. Backend imports are lazy, so
selecting a backend whose extra is not installed fails with an actionable error.
Plain `pip install 'tinytalk[backend]'` does not consume uv's project-level
metadata overrides and is therefore not the qualified installation path while
upstream NeuTTS/OmniVoice metadata still pins the older Torch runtime.

TinyTalk targets Python 3.13 with Torch/Torchaudio 2.11 and Transformers 5.17
across the qualified backend environments.

## API

```
GET  /health
POST /v1/audio/speech
```

`/v1/audio/speech` accepts OpenAI-style JSON. `response_format` may be `wav`
(default), `mp3`, or `opus` (returned as Ogg-Opus). `model` and `voice` are
accepted but currently ignored. `stream: true` is rejected. Long inputs are
chunked at sentence/phrase boundaries server-side.

Backend-specific controls:

- NeuTTS currently ignores `instructions` and `speed` and uses its configured
  reference voice and sampling settings.
- TinyTAuK maps `instructions` to AuK's natural-language voice/style description.
  `speed` scales TinyTalk's target-duration estimate.
- OmniVoice maps `instructions` directly to upstream voice-design `instruct` and
  maps `speed` directly to OmniVoice. With no instructions it uses the configured
  clone prompt when present; otherwise it uses OmniVoice auto-voice mode.

OmniVoice voice design is not a freeform natural-language interface. The request
`instructions` value must contain supported comma-separated attributes, for
example:

```text
male, middle-aged, moderate pitch
female, young adult, british accent
male, elderly, low pitch, whisper
```

The supported English attribute categories are gender (`male`, `female`), age
(`child`, `teenager`, `young adult`, `middle-aged`, `elderly`), pitch (`very low
pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`),
`whisper`, and a fixed set of upstream English accent tags. OmniVoice also
supports its fixed Chinese dialect tags. Unsupported prose or mutually exclusive
attributes raise an upstream validation error rather than being interpreted as a
semantic prompt.

The accepted request `speed` range is `0.25` to `4.0`.

All backends use the same speech-quality evaluator. When `werEndpoint` is set,
TinyTalk transcribes candidate audio through an OpenAI-compatible transcription
endpoint and scores WER/CER before accepting or rerolling a chunk. With no
endpoint configured, the legacy local confidence heuristic is used instead.
OmniVoice starts retries from its greedy default and raises `class_temperature`
on later attempts to introduce controlled variation without changing first-pass
behavior.

Response headers include `X-TinyTalk-Backend`, `X-TinyTalk-Model`,
`X-TinyTalk-Chunks`, `X-TinyTalk-Chunk-Chars`, and `X-TinyTalk-Format`.

## NixOS module

```nix
inputs.tinytalk.url = "github:datGryphon/tinytalk";
```

Import `tinytalk.nixosModules.default` and configure `services.tinytalk`.
NeuTTS remains the default.

The NixOS module separates environment setup from the long-running server.
`tinytalk-setup.service` copies the immutable flake source into
`/var/lib/tinytalk/app`, performs a fresh `uv sync --frozen` there, and keeps
the shared uv download cache under `/var/lib/tinytalk/.cache/uv`. The
`tinytalk.service` unit then runs only the prepared `.venv`; normal process
restarts do not invoke uv or repeat package downloads. If setup fails, the
previous complete app directory is restored.

```nix
services.tinytalk = {
  enable = true;
  backend = "neutts";
};
```

TinyTAuK with the released v0.2 runtime profile:

```nix
services.tinytalk = {
  enable = true;
  backend = "tinytauk";
  tinytaukProfile = ./tinytauk-cpu.toml;
};
```

Copy [the example CPU profile](examples/tinytauk-cpu.toml) into your NixOS
configuration and adjust its component `device`/`dtype` values, seed, and
thread count. Without `tinytaukProfile`, TinyTalk uses the upstream
`TinyTAuK.from_pretrained()` CPU defaults and the existing
`tinytaukModel`/`tinytaukQwenModel` options. With a profile, its `[model]`
table supplies both model IDs; those two NixOS options are not used.

TinyTAuK v0.2 exposes `fp32`, `bf16`, and `fp16` component precision.
It does **not** expose an INT4/INT8 quantization selector. TinyTalk passes
the profile to TinyTAuK without inventing an unsupported quantization option.
Without NixOS, set `TINYTALK_TINYTAUK_PROFILE=/path/to/profile.toml`.

OmniVoice auto/design mode:

```nix
services.tinytalk = {
  enable = true;
  backend = "omnivoice";
  omnivoiceLanguage = "en"; # optional
};
```

OmniVoice configured voice cloning:

```nix
services.tinytalk = {
  enable = true;
  backend = "omnivoice";
  omnivoiceRefAudio = /path/to/reference.wav;
  omnivoiceRefText = /path/to/reference.txt;
};
```

`omnivoiceRefAudio` and `omnivoiceRefText` must be configured together. TinyTalk
creates one reusable OmniVoice clone prompt during engine load and reuses it for
all chunks. A request with `instructions` intentionally selects voice-design mode
for that request instead of the configured clone.

Key options:

| Option | Default | Notes |
| --- | --- | --- |
| `backend` | `neutts` | `neutts`, `tinytauk`, or `omnivoice` |
| `maxCharsPerChunk` | `180` | Max chars per synthesis call |
| `interChunkSilenceMs` | `60` | Silence between chunks |
| `maxRetries` | `2` | Quality-gated retries per chunk |
| `werEndpoint` | empty | OpenAI-compatible transcription base URL |
| `werThreshold` | `0.25` | Shared WER/CER acceptance threshold |
| `model` | `neuphonic/neutts-nano-q4-gguf` | NeuTTS backbone |
| `codec` | `neuphonic/neucodec-onnx-decoder-int8` | NeuTTS codec |
| `backboneDevice` | `cpu` | NeuTTS `cpu` or `gpu` |
| `refCodes` | `/var/lib/tinytalk/ref_codes.pt` | NeuTTS reference codes |
| `refText` | `/var/lib/tinytalk/ref_text.txt` | NeuTTS reference transcript |
| `tinytaukModel` | `tencent/AuK-Flash` | TinyTAuK model repository |
| `tinytaukQwenModel` | `Qwen/Qwen2.5-Omni-3B` | TinyTAuK conditioner (without profile) |
| `tinytaukProfile` | `null` | Optional TinyTAuK v0.2 TOML runtime profile |
| `tinytaukCharsPerSecond` | `14.0` | TinyTAuK duration estimate |
| `omnivoiceModel` | `k2-fsa/OmniVoice` | OmniVoice checkpoint |
| `omnivoiceDevice` | `cpu` | Device map passed upstream |
| `omnivoiceLanguage` | empty | Optional language code/name |
| `omnivoiceRefAudio` | `null` | Optional clone reference audio |
| `omnivoiceRefText` | `null` | Transcript file for clone reference |

NeuTTS reference files are required only for NeuTTS. Generate `ref_codes.pt` from
a WAV with `scripts/encode_reference.py`.

TinyTAuK v0.2.0 downloads the configured AuK-Flash/Qwen checkpoints through
Hugging Face on first startup. TinyTalk performs one disposable warmup generation
before `/health` reports ready.

OmniVoice downloads its checkpoint on first startup. CPU is the conservative
default for TinyTalk; other upstream-supported device strings can be supplied via
`omnivoiceDevice` and should be qualified on the target host before deployment.

An earlier September 2026 qualification run on one CPU host used Python 3.13.13,
OmniVoice 0.2.1, Torch 2.8.0+cpu, and Transformers 5.17.0. The current qualified
development/CI baseline instead uses Torch/Torchaudio 2.11.0 and Transformers 5.17.0. Auto/design generation
used approximately 2.6-2.8 GiB peak process RSS and took roughly 35-53 seconds
for short samples. Cloned generation used approximately 5.1 GiB peak process RSS
and took roughly 100 seconds for similar short samples. Clone-prompt construction
itself took about two seconds. These figures describe that host and corpus only;
they are capacity-planning reference points, not runtime guarantees.

### NeuTTS CPU vs CUDA

The checked-in lock is the qualified CPU runtime and resolves Torch/Torchaudio
from the PyTorch CPU index. `backboneDevice = "gpu"` still controls the NeuTTS
GGUF backbone, but switching the Python runtime itself to CUDA now requires a
separate lock/profile change rather than NixOS-only package overrides. The
NeuCodec pipeline stays on CPU.

## Releases

Releases keep the version bump behind the normal pull-request policy. Run the
**Prepare release** workflow with an `X.Y.Z` version; it creates a
`release/vX.Y.Z` branch, updates `tinytalk.__version__`, refreshes
`uv.lock`, and opens a release PR. After that PR passes CI and is merged, the
**Publish release** workflow tags that exact `main` commit as `vX.Y.Z` and
creates the GitHub release with generated notes. The publish workflow is
idempotent, so rerunning it will not move an existing tag.

## Development

All shells use Python 3.13 and run `uv sync --frozen` against the checked-in
lock. They retain separate virtual environments for backend-aware tests; CI also
checks that all three backends can be synced and tested together from the same
locked dependency graph.

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

Real NeuTTS integration:

```bash
TINYTALK_RUN_INTEGRATION=1 pytest tests/integration/test_real_speech.py
```

Real TinyTAuK integration:

```bash
TINYTALK_RUN_TINYTAUK_INTEGRATION=1 \
pytest tests/integration/test_tinytauk_real_speech.py
```

Real OmniVoice voice-design smoke:

```bash
TINYTALK_RUN_OMNIVOICE_INTEGRATION=1 \
pytest tests/integration/test_omnivoice_real_speech.py
```

Set `TINYTALK_WER_ENDPOINT` during real integration runs when the live shared
WER/CER transcription path also needs validation. Artifacts are written under
`test_artifacts/`.
