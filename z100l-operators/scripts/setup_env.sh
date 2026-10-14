#!/usr/bin/env bash
# Source inside the contest DTK POD.
export H3_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORK_HOME="${WORK_HOME:-$H3_TREE/.runtime}"
mkdir -p "$WORK_HOME"
if [ ! -f /opt/dtk/env.sh ]; then echo 'Requires contest DTK 26.04 POD (/opt/dtk/env.sh)' >&2; return 1; fi
source /opt/dtk/env.sh
export XMAKE_ROOT=y
export INFINI_ROOT="${INFINI_ROOT:-/root/.infini}"
export INFINICORE_DIR="$H3_TREE/InfiniCore"
export PATH="/root/.local/bin:/opt/dtk/cuda/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$H3_TREE/ninetoothed/src:$H3_TREE/tools:${PYTHONPATH:-}"
export HYGON_ARCH=gfx906 H3_DTK_TRITON_COMPAT=1 H3_GFX906_PORTABLE_BF16=1
export TRITON_LIBHIP_PATH=/opt/dtk/lib/libamdhip64.so
