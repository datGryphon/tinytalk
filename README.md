# tinytalk

OpenAI-compatible TTS server backed by NeuTTS designed for low VRAM
environments.

The flake exports `nixosModules.default`.

## API

```
GET  /health
POST /v1/audio/speech
```

`/v1/audio/speech` accepts OpenAI-style JSON. `response_format` may be `wav`
(default), `mp3`, or `opus` (returned as Ogg-Opus). `model`, `voice`, and
`speed` are accepted but ignored. The server uses its configured backbone and
reference voice. `stream: true` is rejected. Long inputs are chunked at
sentence/phrase boundaries server-side.

Response headers: `X-TinyTalk-Chunks`, `X-TinyTalk-Chunk-Chars`,
`X-TinyTalk-Model`, `X-TinyTalk-Format`.

## NixOS module

```nix
inputs.tinytalk.url = "github:datGryphon/tinytalk";
```

Import `tinytalk.nixosModules.default` and configure `services.tinytalk`. Key
options:

| Option                | Default                                | Notes                          |
| --------------------- | -------------------------------------- | ------------------------------ |
| `model`               | `neuphonic/neutts-nano-q4-gguf`        | HF repo or local GGUF path     |
| `codec`               | `neuphonic/neucodec-onnx-decoder-int8` | HF repo or ONNX path           |
| `backboneDevice`      | `cpu`                                  | `cpu` or `gpu`                 |
| `refCodes`            | `/var/lib/tinytalk/ref_codes.pt`       | Pre-encoded reference codes    |
| `refText`             | `/var/lib/tinytalk/ref_text.txt`       | Reference transcript           |
| `port`                | `9002`                                 | uvicorn bind port              |
| `maxCharsPerChunk`    | `180`                                  | Max chars per synthesis call   |
| `interChunkSilenceMs` | `60`                                   | Silence padding between chunks |

The module does not create the reference files. Provide them via tmpfiles or
activation before first start.

To generate `ref_codes.pt` from a WAV file, see `scripts/encode_reference.py`
(adapted from the upstream [neuphonic/neutts-air encode_reference
example](https://huggingface.co/neuphonic/neutts-air)). The encoder requires
`neucodec`, `librosa`, and `torch` but is not needed at serve time.

### CPU vs CUDA

The default runtime installs `torch+cpu` from the PyTorch CPU wheel index. No
GPU required.

For CUDA, set `backboneDevice = "gpu"` and override `runtimeIndexUrl`,
`runtimeExtraIndexUrls`, and `runtimePackages` to pull a CUDA-enabled
`llama-cpp-python` wheel alongside `torch+cpu`. The NeuCodec pipeline stays on
CPU. Only the GGUF backbone offloads to the GPU. On NixOS you will also need to
extend `LD_LIBRARY_PATH` on the service to include the NixOS NVIDIA driver path
(`/run/opengl-driver/lib`) and the CUDA libs installed by the runtime
bootstrap.

## Tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
pytest
```

Real audio integration tests (require model downloads and configured backend):

```bash
TINYTALK_RUN_INTEGRATION=1 \
TINYTALK_REF_CODES=/var/lib/tinytalk/ref_codes.pt \
TINYTALK_REF_TEXT=/var/lib/tinytalk/ref_text.txt \
TINYTALK_BACKBONE_DEVICE=cpu \
pytest tests/integration
```

Artifacts are written under `test_artifacts/`.

Sample in tests/voice taken from the [NeuTTS samples](https://github.com/neuphonic/neutts/tree/main/samples).
