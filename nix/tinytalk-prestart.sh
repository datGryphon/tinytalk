#!/usr/bin/env bash
set -euo pipefail

backend="${TINYTALK_BACKEND:-neutts}"

if [ "$backend" = "neutts" ]; then
  for path in "${TINYTALK_REF_CODES:?}" "${TINYTALK_REF_TEXT:?}"; do
    [ -r "$path" ] || { echo "tinytalk: unreadable: $path" >&2; exit 1; }
  done
fi

if [ "$backend" = "omnivoice" ]; then
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

if [ "$backend" = "tinytauk" ] && [ -n "${TINYTALK_TINYTAUK_PROFILE:-}" ]; then
  [ -r "$TINYTALK_TINYTAUK_PROFILE" ] || {
    echo "tinytalk: unreadable TinyTAuK profile: $TINYTALK_TINYTAUK_PROFILE" >&2
    exit 1
  }
fi

export UV_PROJECT_ENVIRONMENT="${TINYTALK_PYTHON_ENVIRONMENT:?}"

@uv@/bin/uv sync \
  --project "${TINYTALK_PROJECT_ROOT:?}" \
  --frozen \
  --no-dev \
  --python @python@/bin/python \
  --extra "$backend"
