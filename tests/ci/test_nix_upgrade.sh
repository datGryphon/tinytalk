#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
head_commit="$(git rev-parse HEAD)"
previous_tag=""

while IFS= read -r tag; do
  [ -n "$tag" ] || continue
  if [ "$(git rev-parse "${tag}^{commit}")" != "$head_commit" ]; then
    previous_tag="$tag"
    break
  fi
done < <(git tag --merged HEAD --list 'v[0-9]*' --sort=-version:refname)

if [ -z "$previous_tag" ]; then
  echo "tinytalk upgrade test: no previous release tag is available" >&2
  exit 1
fi

tmp="$(mktemp -d)"
cleanup() {
  chmod -R u+w "$tmp" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

previous_source="$tmp/previous-source"
current_source="$tmp/current-source"
broken_source="$tmp/broken-source"
state_dir="$tmp/state"
cache_dir="$tmp/uv-cache"

mkdir -p "$previous_source" "$current_source" "$state_dir" "$cache_dir"

git -C "$repo_root" archive "$previous_tag" | tar -x -C "$previous_source"
git -C "$repo_root" archive HEAD | tar -x -C "$current_source"

# Reproduce the state left by the pre-setup NixOS module: the previous release
# is installed as an editable project into /var/lib/tinytalk/python.
(
  cd "$previous_source"
  UV_PROJECT_ENVIRONMENT="$state_dir/python" \
  UV_CACHE_DIR="$cache_dir" \
    uv sync \
      --frozen \
      --no-dev \
      --python 3.13 \
      --extra neutts
)

previous_version="$(
  "$state_dir/python/bin/python" -I -c 'import tinytalk; print(tinytalk.__version__)'
)"

# A flake source is immutable. Making the current archive read-only ensures the
# setup path cannot accidentally regress to building the editable project there.
chmod -R a-w "$current_source"

TINYTALK_BACKEND=neutts \
TINYTALK_SOURCE="$current_source" \
TINYTALK_APP_DIR="$state_dir/app" \
HOME="$state_dir" \
UV_CACHE_DIR="$cache_dir" \
  "$repo_root/nix/tinytalk-setup.sh"

current_version="$(
  "$state_dir/app/.venv/bin/python" -I -c 'import tinytalk; print(tinytalk.__version__)'
)"
expected_version="$(
  sed -n 's/^__version__ = "\(.*\)"/\1/p' "$current_source/tinytalk/__init__.py"
)"

[ "$current_version" = "$expected_version" ] || {
  echo "tinytalk upgrade test: expected $expected_version, got $current_version" >&2
  exit 1
}

# The current app tree must actually replace the previous release source, even
# on feature branches where the package version has not been bumped yet.
cmp "$current_source/nix/tinytalk-setup.sh" "$state_dir/app/nix/tinytalk-setup.sh"
[ ! -e "$state_dir/app/nix/tinytalk-prestart.sh" ]

# Keep the legacy environment intact until the new service has been qualified;
# deploy-rs may still need it if activation rolls back to the previous module.
legacy_version="$(
  "$state_dir/python/bin/python" -I -c 'import tinytalk; print(tinytalk.__version__)'
)"
[ "$legacy_version" = "$previous_version" ]

# A failed later upgrade must restore the last complete app rather than leave a
# partially-synced environment behind.
cp -R --no-preserve=mode,ownership "$current_source"/. "$broken_source"/
chmod u+w "$broken_source/pyproject.toml"
printf '\n[broken\n' >> "$broken_source/pyproject.toml"
chmod -R a-w "$broken_source"

if TINYTALK_BACKEND=neutts \
   TINYTALK_SOURCE="$broken_source" \
   TINYTALK_APP_DIR="$state_dir/app" \
   HOME="$state_dir" \
   UV_CACHE_DIR="$cache_dir" \
     "$repo_root/nix/tinytalk-setup.sh"; then
  echo "tinytalk upgrade test: broken upgrade unexpectedly succeeded" >&2
  exit 1
fi

restored_version="$(
  "$state_dir/app/.venv/bin/python" -I -c 'import tinytalk; print(tinytalk.__version__)'
)"
[ "$restored_version" = "$current_version" ]
cmp "$current_source/nix/tinytalk-setup.sh" "$state_dir/app/nix/tinytalk-setup.sh"
[ ! -e "$state_dir/app.previous" ]

echo "tinytalk upgrade test: $previous_tag -> working tree passed"
