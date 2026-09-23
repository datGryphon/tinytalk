#!/usr/bin/env bash
set -euo pipefail

backend="${1:?Specify neutts, omnivoice, tinytauk, or shared}"
case "$backend" in
  neutts|omnivoice|tinytauk)
    extras="$backend,test"
    ;;
  shared)
    extras="neutts,omnivoice,tinytauk,test"
    ;;
  *)
    echo "Unknown backend: $backend" >&2
    exit 2
    ;;
esac

uv python install 3.13
uv venv --python 3.13 .venv

override_args=()
override_file="nix/overrides/$backend.txt"
if [ "$backend" = "shared" ]; then
  override_file="nix/overrides/neutts.txt"
fi
if [ -f "$override_file" ]; then
  override_args=(--override "$override_file")
fi

uv pip install \
  --python .venv/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  "${override_args[@]}" \
  -e ".[${extras}]"
