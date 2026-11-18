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
- **Silencio-2** — CPU speech with cached bundled/reference conditioning and
  native inline vocal events and emotions.
- **Chatterbox Nano** — CPU speech with resident bundled/reference conditioning,
  native vocal events and upstream PerTh watermarking.

The flake exports `nixosModules.default`.

## Backend model licensing

TinyTalk itself is MIT-licensed. Backend packages and model weights keep their
own upstream terms; TinyTalk's license does not grant rights to those models.

| Backend | Default model stack | Upstream licensing notes |
| --- | --- | --- |
| **NeuTTS** | [`neuphonic/neutts-nano-q4-gguf`](https://huggingface.co/neuphonic/neutts-nano-q4-gguf) + [`neuphonic/neucodec-onnx-decoder-int8`](https://huggingface.co/neuphonic/neucodec-onnx-decoder-int8) | The default Nano Q4 model repo is gated and currently reports `license: other`; review its bundled `LICENCE` before deployment. The default INT8 NeuCodec repo is Apache-2.0 and also gated. Other selectable NeuTTS checkpoints can differ; for example, [`neuphonic/neutts-air-q4-gguf`](https://huggingface.co/neuphonic/neutts-air-q4-gguf) currently reports Apache-2.0. |
| **TinyTAuK** | [`tencent/AuK-Flash`](https://huggingface.co/tencent/AuK-Flash) + [`Qwen/Qwen2.5-Omni-3B`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | AuK-Flash is MIT-licensed. The default Qwen conditioner uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE), which grants non-commercial use only unless you obtain a separate commercial license from Alibaba Cloud. |
| **OmniVoice** | [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) | OmniVoice code is Apache-2.0, while the published pretrained weights are CC-BY-NC/non-commercial. |
| **Silencio-2** | [`BlueRoseLabs/Silencio-2`](https://huggingface.co/BlueRoseLabs/Silencio-2) | Model weights are CC-BY-4.0; the Pocket TTS core code is MIT. Retain upstream attribution and consult the [upstream NOTICE](https://huggingface.co/BlueRoseLabs/Silencio-2/blob/main/NOTICE.md). |
| **Chatterbox Nano** | [`ResembleAI/chatterbox-nano`](https://huggingface.co/ResembleAI/chatterbox-nano) | Model weights and [Chatterbox code](https://github.com/resemble-ai/chatterbox/blob/ab55923c27757c90cfcc407baa57dc25a7d25217/LICENSE) are MIT-licensed; retain upstream notices. |

These notes describe the default model IDs in this repository as of November
2026. If you point TinyTalk at different checkpoints or a custom TinyTAuK
profile, check the licenses for the complete model stack you actually deploy.

## Python environments

`pyproject.toml` and `uv.lock` are the source of truth for TinyTalk's Python
runtime. The project pins the qualified Torch/Torchaudio/Transformers versions,
the PyTorch CPU index, the NeuCodec fork, and backend package versions once.
Development, CI, and the NixOS service all consume that same lock.

From a source checkout, install the shared development environment:

```bash
uv sync --frozen --all-extras
```

All five backends and the tests coexist in one `.venv`. Select the resident
backend with `TINYTALK_BACKEND` (default `neutts`); changing it does not require
another development environment.

For minimal deployments, the NixOS service installs only the selected backend's
extra into its single persistent `.venv`. Backend imports are lazy, so
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
(default), `mp3`, or `opus` (returned as Ogg-Opus). `model` is accepted but
ignored. With named voices configured,
`voice` selects a preset; when omitted the configured default is used. Unknown
voice names return HTTP 400. Without named voices, `voice` retains its legacy
ignored behavior. `stream: true` is rejected. Long inputs are
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

NeuTTS, TinyTAuK and OmniVoice use the same speech-quality evaluator. When `werEndpoint` is set,
TinyTalk transcribes candidate audio through an OpenAI-compatible transcription
endpoint and scores WER/CER before accepting or rerolling a chunk. With no
endpoint configured, the legacy local confidence heuristic is used instead.
OmniVoice starts retries from its greedy default and raises `class_temperature`
on later attempts to introduce controlled variation without changing first-pass
behavior.

Response headers include `X-TinyTalk-Backend`, `X-TinyTalk-Model`,
`X-TinyTalk-Chunks`, `X-TinyTalk-Chunk-Chars`, and `X-TinyTalk-Format`.

## Named voices

A service still loads one backend/model. Named voices select static conditioning
for that backend, not a different model. Define `TINYTALK_VOICES_FILE` as a
path to a JSON file, or use the NixOS module options below. The file is validated
at startup. The backend must match `TINYTALK_BACKEND`, and `default` must name
one of the configured voices.

Example for TinyTAuK:

```json
{
  "backend": "tinytauk",
  "default": "narrator",
  "voices": {
    "narrator": {"description": "A clear, natural speaking voice"},
    "dramatic": {
      "description": "A dramatic, expressive voice",
      "max_chars_per_chunk": 120,
      "inter_chunk_silence_ms": 100
    }
  }
}
```

The equivalent NixOS configuration:

```nix
services.tinytalk = {
  enable = true;
  backend = "tinytauk";
  defaultVoice = "narrator";
  voices = {
    narrator.description = "A clear, natural speaking voice";
    dramatic = {
      description = "A dramatic, expressive voice";
      maxCharsPerChunk = 120;
      interChunkSilenceMs = 100;
    };
  };
};
```

Supported preset fields depend on the backend:

- NeuTTS: `ref_codes` and `ref_text` (both required).
- OmniVoice: `ref_audio` and `ref_text` (both required).
- Silencio-2: `voice_id` (bundled ID or reference-audio path).
- Chatterbox Nano: `ref_audio` (required).
- TinyTAuK: `description` (optional; request `instructions` take precedence).

All backends support optional `max_chars_per_chunk` and
`inter_chunk_silence_ms` per voice. Nix uses camelCase names as in the
example. Reference files must be readable by the `tinytalk` service user.
The JSON configuration contains reference *paths*, not the audio contents.
Model loading remains single-instance; reference conditioning is prepared at
startup where supported. Named voices replace the service's single-reference
settings. OmniVoice request `instructions` still select voice-design mode
instead of cloning; TinyTAuK request `instructions` override the preset
description. No runtime enrollment is provided.

## NixOS module

```nix
inputs.tinytalk.url = "github:datGryphon/tinytalk";
```

Import `tinytalk.nixosModules.default` and configure `services.tinytalk`.
NeuTTS remains the default.

The NixOS service links its immutable flake source into
`/var/lib/tinytalk/app` and runs `uv sync --frozen` there. Ordinary restarts
reuse the installed `.venv` and uv download cache; a source, lockfile, or
backend change updates only what changed.

Installation runs in the service launcher, not `ExecStartPre`, so a changed
environment does not block NixOS activation or deploy-rs confirmation.
`systemctl is-active` can report `active` during installation; use `/health`
to verify inference readiness.

TinyTalk supplies Nix-native CMake, GNU Make, and compiler dependencies.
NeuTTS keeps the runtime device separate from the compiled llama.cpp backend.
The default is CPU. Select `llamaCppBackend = "vulkan"` or `"cuda"` alongside
`backboneDevice = "gpu"`; the NixOS module configures the corresponding source
build toolchain.

Example Vulkan configuration:

```nix
services.tinytalk = {
  enable = true;
  backend = "neutts";
  backboneDevice = "gpu";
  llamaCppBackend = "vulkan";
};
```

For NVIDIA GPUs, select `llamaCppBackend = "cuda"` and optionally set
`llamaCppCudaArchitectures` to the target GPU's compute capability (for
example, `"61"` for Pascal). CUDA source builds use Nixpkgs CUDA 12 and
the locked `llama-cpp-python==0.3.35` with Python 3.13. This path is
**experimental and not validated by a CUDA runtime test**; confirm the GPU,
driver, toolkit, and CPU compatibility before deployment. Nixpkgs CUDA
packages require `allowUnfree`.

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
| `backend` | `neutts` | `neutts`, `tinytauk`, `omnivoice`, `silencio2`, or `chatterbox_nano` |
| `voices` | `{}` | Static named voice presets for the selected backend |
| `defaultVoice` | `null` | Required default name when `voices` is nonempty |
| `maxCharsPerChunk` | `180` | Max chars per synthesis call |
| `interChunkSilenceMs` | `60` | Silence between chunks |
| `maxRetries` | `2` | Quality-gated retries per chunk |
| `werEndpoint` | empty | OpenAI-compatible transcription base URL |
| `werThreshold` | `0.25` | Shared WER/CER acceptance threshold |
| `model` | `neuphonic/neutts-nano-q4-gguf` | NeuTTS backbone |
| `codec` | `neuphonic/neucodec-onnx-decoder-int8` | NeuTTS codec |
| `backboneDevice` | `cpu` | NeuTTS runtime `cpu` or `gpu` |
| `llamaCppBackend` | `cpu` | NeuTTS build `cpu`, `vulkan`, `cuda` |
| `llamaCppCudaArchitectures` | `null` | Optional CUDA architecture (e.g. `61`) |
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

NeuTTS requires reference files for either its single configured voice or each
named voice. Generate `ref_codes.pt` from a WAV with `scripts/encode_reference.py`.

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

Python Semantic Release runs after CI succeeds on pushes to `main`.
Conventional commits determine the SemVer bump: `feat` is minor,
`fix`/`perf` are patch, and a breaking change is major. `pyproject.toml`
owns the package version; `tinytalk.__version__` reads the installed package
metadata. A release updates `uv.lock`, writes the changelog, commits the version,
tags `vX.Y.Z`, and creates the GitHub release. PyPI publishing is not configured.

## Development

The development shell uses Python 3.13 and syncs all five backends and tests
into one `.venv` from the checked-in lock. Native and Nix CI each run the full
unit suite once in that combined environment.

```bash
nix develop
pytest tests --ignore=tests/integration
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
## Silencio-2 CPU Backend

Select `TINYTALK_BACKEND=silencio2` in the shared environment.
`TINYTALK_SILENCIO2_VOICE` selects an upstream bundled ID (default `260`) or a
local reference-audio path. NixOS exposes `services.tinytalk.silencio2Voice`.
Without named voices, this configured voice is encoded at startup and reused.
With named voices, each preset's `voice_id` is prepared and selected by HTTP `voice`.

TinyTalk loads the original `silencio2` Python package directly from the pinned
Hugging Face snapshot `20930f5f6a6bb564f2a8207152fd5c5d8eecff03`. Upstream does
not publish an installable Python distribution, so the optional extra declares
its inference dependencies, while the snapshot supplies code, weights and bundled
reference clips. Initial startup needs network/cache access to Hugging Face.
No upstream source is copied into TinyTalk.

Native sounds (`[laugh]`, `[sigh]`, `[gasp]`, `[cough]`, `[clear_throat]`) and
delivery tags (`<happy>`, `<sad>`, `<angry>`, `<fear>`, `<surprised>`, `<whisper>`)
belong in `input`, including with a configured reference. Nonempty `instructions`
and speed other than `1.0` are unsupported. Output retains upstream's native
24 kHz audio without loudness normalization or TinyTalk quality retries.
The shared tagged-input splitter keeps tags atomic; active Silencio emotions
carry across TinyTalk chunks. A repeated emotion prefix can add a few characters
to the configured text budget.

Silencio-2 weights are CC-BY-4.0, based on Kyutai Pocket TTS; its core code is
MIT. Retain upstream attribution and notices when redistributing model assets:
<https://huggingface.co/BlueRoseLabs/Silencio-2/blob/main/NOTICE.md>.

## Chatterbox Nano CPU Backend

Select `TINYTALK_BACKEND=chatterbox_nano` in the shared environment. The optional
`TINYTALK_CHATTERBOX_NANO_REF_AUDIO` (`services.tinytalk.chatterboxNanoRefAudio`)
selects a local reference clip longer than five seconds. With no reference,
upstream's bundled conditioning is used. With named voices, each preset's
`ref_audio` is prepared once and selected by HTTP `voice`.

TinyTalk pins upstream Chatterbox source to
`ab55923c27757c90cfcc407baa57dc25a7d25217`; the released 0.1.7 wheel lacks
`nano=True`. The adapter calls `ChatterboxTurboTTS.from_pretrained(device="cpu",
nano=True)` and encodes a configured reference once during load. Calls then reuse
that conditioning without reference-file reads. Reference normalization is
disabled through upstream's public `norm_loudness=False` option, avoiding the
observed normalization dtype mismatch and preserving the supplied clip's level.

This backend is verified on TinyTalk's Python 3.13, Torch/Torchaudio 2.11 and
Transformers 5.17 stack. Dependency overrides retain existing Perth 1.0.1 and
allow Safetensors >=0.8 as required by Transformers, rather than installing a
second ML stack. Upstream generation, vocal-event tags and watermarking remain
unchanged; there are no TinyTalk DSP transforms or quality retries. Put tags such
as `[laugh]` in `input`; nonempty `instructions` and non-unit speed are unsupported.

Chatterbox code/model licensing is MIT; consult upstream asset notices. The pinned
upstream README declares the research repository unmaintained. Its full Nano
snapshot is approximately 3 GB and qualification used approximately 4.2 GB peak
process memory; these are development-host observations, not capacity guarantees.
