#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
python -c 'import sys,torch; assert sys.version_info[:2]==(3,11), "Python 3.11 required"; assert torch.version.hip, "DTK HIP PyTorch required"; assert torch.cuda.is_available(), "DCU required"'
PIP_SOURCE=()
if [ -n "${DEPS_DIR:-}" ]; then PIP_SOURCE=(--no-index --find-links "$DEPS_DIR"); fi
for item in triton:3.8.0 pybind11:2.13.6; do
  package=${item%%:*}; version=${item#*:}
  if ! python -c "import importlib.metadata as m; assert m.version('$package') == '$version'" 2>/dev/null; then
    python -m pip install "${PIP_SOURCE[@]}" --no-deps "$package==$version"
  fi
done
python -c 'import numpy,sympy'
