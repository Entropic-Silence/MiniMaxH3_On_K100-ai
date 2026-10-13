#!/usr/bin/env bash
set -euo pipefail
H3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
set +u
source "${DTK_ENV:-/opt/dtk/env.sh}"
set -u
export INFINI_ROOT=${INFINI_ROOT:-"$H3_ROOT/.infini"}
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export INFINICORE_HYGON_MHA=${INFINICORE_HYGON_MHA:-aten}
exec "${PYTHON:-python}" "$H3_ROOT/scripts/run_case.py" "$@"
