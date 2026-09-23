#!/usr/bin/env bash
set -euo pipefail

PYTHON_TARGET="${TINYTALK_PYTHON_TARGET:?}"
MARKER="$PYTHON_TARGET/.spec"
INDEX_STRATEGY="unsafe-best-match"
SPEC="v5|@python@|${TINYTALK_BACKEND:-neutts}|$INDEX_STRATEGY|${TINYTALK_PIP_INDEX_URL:-}|${TINYTALK_PIP_EXTRA_INDEX_URLS:-}|${TINYTALK_RUNTIME_REQUIREMENTS:?}|${TINYTALK_RUNTIME_OVERRIDE:-}"

if [ "${TINYTALK_BACKEND:-neutts}" = "neutts" ]; then
  for path in "${TINYTALK_REF_CODES:?}" "${TINYTALK_REF_TEXT:?}"; do
    [ -r "$path" ] || { echo "tinytalk: unreadable: $path" >&2; exit 1; }
  done
fi

if [ "${TINYTALK_BACKEND:-neutts}" = "omnivoice" ]; then
  ref_audio="${TINYTALK_OMNIVOICE_REF_AUDIO:-}"
  ref_text="${TINYTALK_OMNIVOICE_REF_TEXT:-}"
  if [ -n "$ref_audio" ] || [ -n "$ref_text" ]; then
    [ -n "$ref_audio" ] && [ -n "$ref_text" ] || {
      echo "tinytalk: OmniVoice reference audio and text must be configured together" >&2
      exit 1
    }
    for path in "$ref_audio" "$ref_text"; do
      [ -r "$path" ] || { echo "tinytalk: unreadable: $path" >&2; exit 1; }
    done
  fi
fi

if [ "${TINYTALK_BACKEND:-neutts}" = "tinytauk" ] && [ -n "${TINYTALK_TINYTAUK_PROFILE:-}" ]; then
  [ -r "$TINYTALK_TINYTAUK_PROFILE" ] || {
    echo "tinytalk: unreadable TinyTAuK profile: $TINYTALK_TINYTAUK_PROFILE" >&2
    exit 1
  }
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

OVERRIDE=()
if [ -n "${TINYTALK_RUNTIME_OVERRIDE:-}" ]; then
  OVERRIDE=(--override "$TINYTALK_RUNTIME_OVERRIDE")
fi

@uv@/bin/uv pip install \
  --python @python@/bin/python \
  --target "$PYTHON_TARGET" \
  --index-strategy "$INDEX_STRATEGY" \
  ${TINYTALK_PIP_INDEX_URL:+--index-url "$TINYTALK_PIP_INDEX_URL"} \
  "${EXTRA[@]}" \
  "${OVERRIDE[@]}" \
  -r "$TINYTALK_RUNTIME_REQUIREMENTS"

printf '%s' "$SPEC" > "$MARKER"
