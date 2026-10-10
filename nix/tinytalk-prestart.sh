#!/usr/bin/env bash
set -euo pipefail

source_root="${1:?source path required}"
app="${2:-/var/lib/tinytalk/app}"
backend="${TINYTALK_BACKEND:?}"
stamp="$app/.tinytalk-installed"
revision="$source_root:$backend:${TINYTALK_LLAMA_CPP_BACKEND:-none}"

# Reuse successful installations on ordinary restarts. A source/backend change
# or interrupted install starts clean; the uv download cache is retained.
if [[ -x "$app/.venv/bin/python" && -f "$stamp" && "$(< "$stamp")" == "$revision" ]]; then
  exit 0
fi

rm -rf "$app"
mkdir -p "$app"
cp -R --no-preserve=mode,ownership "$source_root"/. "$app"/

cd "$app"
uv sync --frozen --no-dev --python python3 --extra "$backend"
printf '%s\n' "$revision" > "$stamp"
