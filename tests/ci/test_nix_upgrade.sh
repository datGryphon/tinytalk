#!/usr/bin/env bash
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
previous="$(git tag --merged HEAD --list 'v[0-9]*' --sort=-version:refname | head -1)"
[ -n "$previous" ] || { echo "no previous release tag" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'chmod -R u+w "$tmp" 2>/dev/null || true; rm -rf "$tmp"' EXIT

mkdir -p "$tmp/source" "$tmp/app" "$tmp/cache"
git -C "$repo" archive "$previous" | tar -x -C "$tmp/app"
(
  cd "$tmp/app"
  UV_CACHE_DIR="$tmp/cache" uv sync --frozen --no-dev --python 3.13 --extra neutts
)

git -C "$repo" archive HEAD | tar -x -C "$tmp/source"
chmod -R a-w "$tmp/source"

TINYTALK_BACKEND=neutts \
UV_CACHE_DIR="$tmp/cache" \
  "$repo/nix/tinytalk-prestart.sh" "$tmp/source" "$tmp/app"

cmp "$tmp/source/tinytalk/__init__.py" "$tmp/app/tinytalk/__init__.py"
test -x "$tmp/app/.venv/bin/python"
"$tmp/app/.venv/bin/python" -I -c 'import tinytalk'
