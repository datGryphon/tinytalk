# tinytalk

OpenAI-compatible TTS server for modest machines. TinyTalk owns the HTTP API,
chunking, stitching, output encoding, and service lifecycle while synthesis is
provided by a selectable backend.

Supported backends:

- **NeuTTS** — the existing reference-voice/GGUF path;
- **TinyTAuK** — instruction-controlled AuK-Flash through
  [`datGryphon/tinytauk`](https://github.com/datGryphon/tinytauk).

The flake exports `nixosModules.default`.

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
- TinyTAuK maps `instructions` to AuK's natural-language voice/style
  description. `speed` scales TinyTalk's target-duration estimate; `1.0` is the
  configured baseline, with the accepted range `0.25` to `4.0`.

Response headers include `X-TinyTalk-Backend`, `X-TinyTalk-Model`,
`X-TinyTalk-Chunks`, `X-TinyTalk-Chunk-Chars`, and `X-TinyTalk-Format`.

## NixOS module

```nix
inputs.tinytalk.url = "github:datGryphon/tinytalk";
```

Import `tinytalk.nixosModules.default` and configure `services.tinytalk`.
NeuTTS remains the default:

```nix
services.tinytalk = {
  enable = true;
  backend = "neutts";
};
```

For TinyTAuK:

```nix
services.tinytalk = {
  enable = true;
  backend = "tinytauk";
};
```

A service instance runs one backend. This is intentional: TinyTAuK v0.1.0 uses
Python 3.13 / Torch 2.7.1, while current NeuTTS uses a newer Torch/Transformers
stack. The NixOS module selects the matching Python/runtime package set instead
of trying to co-install incompatible ML environments.

Key options:

| Option | Default | Notes |
| --- | --- | --- |
| `backend` | `neutts` | `neutts` or `tinytauk` |
| `model` | `neuphonic/neutts-nano-q4-gguf` | NeuTTS backbone |
| `codec` | `neuphonic/neucodec-onnx-decoder-int8` | NeuTTS codec |
| `backboneDevice` | `cpu` | NeuTTS `cpu` or `gpu` |
| `refCodes` | `/var/lib/tinytalk/ref_codes.pt` | NeuTTS reference codes |
| `refText` | `/var/lib/tinytalk/ref_text.txt` | NeuTTS reference transcript |
| `tinytaukModel` | `tencent/AuK-Flash` | TinyTAuK model repository |
| `tinytaukQwenModel` | `Qwen/Qwen2.5-Omni-3B` | TinyTAuK conditioner |
| `tinytaukCharsPerSecond` | `14.0` | TinyTAuK duration estimate |
| `maxCharsPerChunk` | `180` | Max chars per synthesis call |
| `interChunkSilenceMs` | `60` | Silence between chunks |

NeuTTS reference files are required only for the NeuTTS backend. The module does
not create them. Generate `ref_codes.pt` from a WAV with
`scripts/encode_reference.py`.

TinyTAuK downloads the official AuK-Flash/Qwen checkpoints through Hugging Face
on first startup. Its VAE compiles lazily, so TinyTalk performs one disposable
9-second generation during engine load before `/health` reports ready. The
module uses larger default systemd memory limits for this backend (17 GB soft /
20 GB hard) than for NeuTTS.

### NeuTTS CPU vs CUDA

The default NeuTTS runtime installs `torch+cpu` from the PyTorch CPU wheel
index. For CUDA, set `backboneDevice = "gpu"` and override `runtimeIndexUrl`,
`runtimeExtraIndexUrls`, and `runtimePackages` to pull a CUDA-enabled
`llama-cpp-python` wheel. The NeuCodec pipeline stays on CPU.

## Development

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

The two shells use separate virtual environments because their ML dependency
stacks are intentionally incompatible.

Real NeuTTS integration tests:

```bash
TINYTALK_RUN_INTEGRATION=1 pytest tests/integration/test_real_speech.py
```

Real TinyTAuK smoke test (downloads models on first run and may compile for
several minutes):

```bash
TINYTALK_RUN_TINYTAUK_INTEGRATION=1 \
pytest tests/integration/test_tinytauk_real_speech.py
```

Artifacts are written under `test_artifacts/`.
