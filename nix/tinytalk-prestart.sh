#!/usr/bin/env bash
set -euo pipefail

source_root="${1:?source path required}"
app="${2:-/var/lib/tinytalk/app}"

rm -rf "$app"
mkdir -p "$app"
cp -R --no-preserve=mode,ownership "$source_root"/. "$app"/

cd "$app"
uv sync --frozen --no-dev --python python3 --extra "${TINYTALK_BACKEND:?}"
