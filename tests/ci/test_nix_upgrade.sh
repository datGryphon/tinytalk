#!/usr/bin/env bash
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
previous="$(git tag --merged HEAD --list 'v[0-9]*' --sort=-version:refname | head -1)"
[ -n "$previous" ] || { echo "no previous release tag" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'chmod -R u+w "$tmp" 2>/dev/null || true; rm -rf "$tmp"' EXIT

mkdir -p "$tmp/old" "$tmp/current" "$tmp/app" "$tmp/cache"
git -C "$repo" archive "$previous" | tar -x -C "$tmp/old"
git -C "$repo" archive HEAD | tar -x -C "$tmp/current"

# Start from an installed previous release.
cp -R --no-preserve=mode,ownership "$tmp/old"/. "$tmp/app"/
(
  cd "$tmp/app"
  UV_CACHE_DIR="$tmp/cache" uv sync --frozen --no-dev --python 3.13 --extra neutts
)

# Nix sources are read-only. The new prestart must copy before syncing.
chmod -R a-w "$tmp/current"
touch "$tmp/ref.pt" "$tmp/ref.txt"

TINYTALK_BACKEND=neutts \
TINYTALK_PROJECT_ROOT="$tmp/current" \
TINYTALK_APP_ROOT="$tmp/app" \
TINYTALK_REF_CODES="$tmp/ref.pt" \
TINYTALK_REF_TEXT="$tmp/ref.txt" \
UV_CACHE_DIR="$tmp/cache" \
  "$repo/nix/tinytalk-prestart.sh"

cmp "$tmp/current/tinytalk/__init__.py" "$tmp/app/tinytalk/__init__.py"
test -x "$tmp/app/.venv/bin/python"
"$tmp/app/.venv/bin/python" -I -c 'import tinytalk'

echo "upgrade from $previous passed"
