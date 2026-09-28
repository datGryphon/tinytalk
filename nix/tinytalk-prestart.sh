#!/usr/bin/env bash
set -euo pipefail

app="${TINYTALK_APP_ROOT:?}"

rm -rf "$app"
mkdir -p "$app"
cp -R --no-preserve=mode,ownership "${TINYTALK_SOURCE:?}"/. "$app"/

cd "$app"
uv sync --frozen --no-dev --python python3 --extra "${TINYTALK_BACKEND:?}"
