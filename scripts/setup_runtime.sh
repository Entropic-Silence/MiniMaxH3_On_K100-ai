#!/usr/bin/env bash
# User-approved isolated dependency override. Never install into the base Python.
set -euo pipefail
H3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
H3_BASE_PYTHON=${H3_BASE_PYTHON:-python3}
H3_VENV=${H3_VENV:-"$H3_ROOT/.venv"}
set +u
source "${DTK_ENV:-/opt/dtk/env.sh}"
set -u
mkdir -p "$H3_ROOT/logs/runtime"
H3_SNAPSHOT='import importlib,importlib.metadata as m,json; print(json.dumps({n:{"version":m.version(n),"path":importlib.import_module(n).__file__} for n in ("torch","triton","numpy")},sort_keys=True))'
H3_BASE_BEFORE=$("$H3_BASE_PYTHON" -c "$H3_SNAPSHOT")
if [[ ! -f "$H3_VENV/pyvenv.cfg" ]]; then
  "$H3_BASE_PYTHON" -m venv --system-site-packages --without-pip "$H3_VENV"
fi
"$H3_VENV/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix, "isolated venv required"'
"$H3_VENV/bin/python" -m pip install --ignore-installed --no-index --no-deps --require-hashes \
  --find-links "$H3_ROOT/third_party/wheels" -r "$H3_ROOT/configs/runtime-requirements.txt"
H3_BASE_AFTER=$("$H3_BASE_PYTHON" -c "$H3_SNAPSHOT")
H3_VENV_PROTECTED=$("$H3_VENV/bin/python" -c "$H3_SNAPSHOT")
[[ "$H3_BASE_BEFORE" == "$H3_BASE_AFTER" && "$H3_BASE_BEFORE" == "$H3_VENV_PROTECTED" ]] || {
  echo 'Protected torch/Triton/numpy versions or module paths differ; stopping.' >&2; exit 1;
}
printf '%s\n' "$H3_VENV_PROTECTED" > "$H3_ROOT/logs/runtime/protected-packages.json"
"$H3_VENV/bin/python" "$H3_ROOT/scripts/check_runtime.py" "$@" | tee "$H3_ROOT/logs/runtime/preflight.json"
printf 'Use PYTHON=%s/bin/python with build_all.sh and run_case.sh\n' "$H3_VENV"
