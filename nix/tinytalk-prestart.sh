#!/usr/bin/env bash
set -euo pipefail

PYTHON_TARGET="${TINYTALK_PYTHON_TARGET:?}"
MARKER="$PYTHON_TARGET/.spec"
INDEX_STRATEGY="unsafe-best-match"
SPEC="v3|${TINYTALK_BACKEND:-neutts}|$INDEX_STRATEGY|${TINYTALK_PIP_INDEX_URL:-}|${TINYTALK_PIP_EXTRA_INDEX_URLS:-}|${TINYTALK_RUNTIME_REQUIREMENTS:?}"

if [ "${TINYTALK_BACKEND:-neutts}" = "neutts" ]; then
  for path in "${TINYTALK_REF_CODES:?}" "${TINYTALK_REF_TEXT:?}"; do
    [ -r "$path" ] || { echo "tinytalk: unreadable: $path" >&2; exit 1; }
  done
fi

if [ "$(cat "$MARKER" 2>/dev/null)" = "$SPEC" ]; then
  exit 0
fi

rm -rf "$PYTHON_TARGET"
mkdir -p "$PYTHON_TARGET"

EXTRA=()
for url in ${TINYTALK_PIP_EXTRA_INDEX_URLS:-}; do
  EXTRA+=(--extra-index-url "$url")
done

@uv@/bin/uv pip install \
  --python @python@/bin/python \
  --target "$PYTHON_TARGET" \
  --index-strategy "$INDEX_STRATEGY" \
  ${TINYTALK_PIP_INDEX_URL:+--index-url "$TINYTALK_PIP_INDEX_URL"} \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -r "$TINYTALK_RUNTIME_REQUIREMENTS"

printf '%s' "$SPEC" > "$MARKER"
