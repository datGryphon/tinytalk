#!/usr/bin/env bash
set -euo pipefail

source_root="${1:?source path required}"
app="${2:-/var/lib/tinytalk/app}"

mkdir -p "$app"
for name in pyproject.toml uv.lock tinytalk; do
  rm -rf -- "$app/$name"
  ln -s -- "$source_root/$name" "$app/$name"
done

cd "$app"
uv sync --frozen --no-dev --python python3 --extra "${TINYTALK_BACKEND:?}"
