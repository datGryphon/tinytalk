#!/usr/bin/env bash
set -euo pipefail

# Run inside a native uv venv or the matching Nix backend dev shell.
case "${TINYTALK_BACKEND:-}" in
  neutts|omnivoice|tinytauk) ;;
  *) echo "Set TINYTALK_BACKEND to neutts, omnivoice, or tinytauk" >&2; exit 2 ;;
esac

python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json
import os
import sys

backend = os.environ["TINYTALK_BACKEND"]
assert sys.version_info[:2] == (3, 13), sys.version

expected = {
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "transformers": "5.17.0",
}
if backend == "neutts":
    expected.update(neutts="1.4.1", neucodec="0.0.6")
elif backend == "omnivoice":
    expected["omnivoice"] = "0.2.1"
else:
    expected["tinytauk"] = "0.2.0"

for package, wanted in expected.items():
    actual = metadata.version(package)
    print(f"{package}: {actual}")
    assert actual.split("+")[0] == wanted, (package, actual, wanted)
    if package in ("torch", "torchaudio"):
        assert actual.endswith("+cpu"), (package, actual)

if backend in ("neutts", "tinytauk"):
    for removed in ("torchtune", "torchao"):
        assert importlib.util.find_spec(removed) is None, removed

if backend == "neutts":
    direct_url = metadata.distribution("neucodec").read_text("direct_url.json")
    assert direct_url is not None, "NeuCodec must come from the qualified fork"
    source = json.loads(direct_url)["url"]
    print(f"neucodec source: {source}")
    assert "6954b1f877963e19177b43be1ef56b1818990d31" in source, source
elif backend == "tinytauk":
    direct_url = metadata.distribution("tinytauk").read_text("direct_url.json")
    assert direct_url is not None, "TinyTAuK must come from the v0.2.0 release"
    source = json.loads(direct_url)["url"]
    print(f"tinytauk source: {source}")
    assert "/v0.2.0.tar.gz" in source, source
PY

case "$TINYTALK_BACKEND" in
  neutts)
    python -m pytest tests --ignore=tests/integration -q
    ;;
  omnivoice)
    # Generic API/reroll tests assume a NeuTTS-style synthesize signature.
    python -m pytest tests/test_omnivoice_backend.py tests/test_backend_api.py -q
    ;;
  tinytauk)
    python -m pytest tests/test_tinytauk_backend.py tests/test_backend_api.py -q
    ;;
esac
