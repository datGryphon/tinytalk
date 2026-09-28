#!/usr/bin/env bash
set -euo pipefail

backend="${TINYTALK_BACKEND:?}"
source_root="${TINYTALK_SOURCE:?}"
app_dir="${TINYTALK_APP_DIR:?}"
previous_dir="${app_dir}.previous"

# Recover the last complete deployment if a prior setup was interrupted after
# moving it aside but before the replacement finished.
if [ -e "$previous_dir" ]; then
  rm -rf "$app_dir"
  mv "$previous_dir" "$app_dir"
fi

if [ -e "$app_dir" ]; then
  mv "$app_dir" "$previous_dir"
fi

restore_previous() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    rm -rf "$app_dir"
    if [ -e "$previous_dir" ]; then
      mv "$previous_dir" "$app_dir"
    fi
  fi
  exit "$status"
}
trap restore_previous EXIT

mkdir -p "$app_dir"
cp -R --no-preserve=mode,ownership "$source_root"/. "$app_dir"/

cd "$app_dir"
uv sync \
  --frozen \
  --no-dev \
  --python 3.13 \
  --extra "$backend"

trap - EXIT
rm -rf "$previous_dir"
