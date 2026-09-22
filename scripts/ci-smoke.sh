#!/usr/bin/env bash
set -euo pipefail

# Run inside either the native uv venv or a Nix backend dev shell.
case "${TINYTALK_BACKEND:-}" in
  neutts|omnivoice) ;;
  *) echo "Set TINYTALK_BACKEND to neutts or omnivoice" >&2; exit 2 ;;
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
else:
    expected["omnivoice"] = "0.2.1"

for package, wanted in expected.items():
    actual = metadata.version(package)
    print(f"{package}: {actual}")
    assert actual.split("+")[0] == wanted, (package, actual, wanted)
    if package in ("torch", "torchaudio"):
        assert actual.endswith("+cpu"), (package, actual)

if backend == "neutts":
    for removed in ("torchtune", "torchao"):
        assert importlib.util.find_spec(removed) is None, removed

    direct_url = metadata.distribution("neucodec").read_text("direct_url.json")
    assert direct_url is not None, "NeuCodec must come from the qualified fork"
    source = json.loads(direct_url)["url"]
    print(f"neucodec source: {source}")
    assert "6954b1f877963e19177b43be1ef56b1818990d31" in source, source
PY

if [[ "$TINYTALK_BACKEND" == "neutts" ]]; then
  python -m pytest tests --ignore=tests/integration -q
else
  # The generic API/reroll tests assume a NeuTTS-style synthesize signature.
  python -m pytest tests/test_omnivoice_backend.py tests/test_backend_api.py -q
fi
