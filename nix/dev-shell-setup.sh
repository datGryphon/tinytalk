# Sourced by the development shells. Do not change the caller's shell options.
expected_python="$("$TINYTALK_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
current_python=""

if [ -x "$TINYTALK_VENV/bin/python" ]; then
  current_python="$("$TINYTALK_VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
fi

if [ "$current_python" != "$expected_python" ]; then
  rm -rf "$TINYTALK_VENV"
  uv venv "$TINYTALK_VENV" --python "$TINYTALK_PYTHON" --python-preference only-system || return 1
fi

override_args=()
if [ -n "$TINYTALK_OVERRIDE_FILE" ]; then
  override_args=(--override "$TINYTALK_OVERRIDE_FILE")
fi

uv pip install \
  --python "$TINYTALK_VENV/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  "${override_args[@]}" \
  -e ".[${TINYTALK_BACKEND},test]" || return 1

source "$TINYTALK_VENV/bin/activate"
