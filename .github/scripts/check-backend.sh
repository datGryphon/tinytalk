#!/usr/bin/env bash
set -euo pipefail

backend="${TINYTALK_BACKEND:?Set TINYTALK_BACKEND}"
python .github/scripts/check-runtime.py "$backend"

case "$backend" in
  neutts)
    python -m pytest tests --ignore=tests/integration -q
    ;;
  omnivoice)
    # The generic reroll tests depend on a NeuTTS-shaped test double.
    python -m pytest tests/test_omnivoice_backend.py tests/test_backend_api.py -q
    ;;
  tinytauk)
    python -m pytest tests/test_tinytauk_backend.py tests/test_backend_api.py -q
    ;;
  *)
    echo "Unsupported backend: $backend" >&2
    exit 2
    ;;
esac
