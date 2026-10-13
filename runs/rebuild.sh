#!/bin/bash
# incremental rebuild of InfiniCore (+_infinicore) and InfiniLM on the node; usage: rebuild_remote.sh [core|lm|all|nt]
# nt: regenerate the NineToothed HIP AOT kernels (scripts/build_h3_attention.py) first, then core + lm
set -o pipefail
cd "${H3_ROOT:-/root/h3-work/h3-extreme}" && source logs/k100/env.sh >/dev/null 2>&1
export XMAKE_ROOT=y H3_OFFLINE_BUILD=1 PATH=$PWD/.local/bin:$PATH HYGON_CUDA_HOME=/opt/dtk/cuda/cuda
export H3_PYBIND11_INCLUDE=$($PYTHON -c 'import pybind11; print(pybind11.get_include())')
export H3_PYTHON_INCLUDE=$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
export INFINILM_CXX11_ABI=$($PYTHON -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
what=${1:-all}
log=/tmp/rebuild.log; : > $log
step() { echo "== $*" >> $log; "$@" >> $log 2>&1 || { echo "FAILED: $*"; grep -aE "error" $log | head -20; exit 1; }; }
if [ "$what" = nt ]; then
  step env PYTHONPATH=$PWD/ninetoothed/src:$PWD/tools $PYTHON scripts/build_h3_attention.py
  what=all
fi
if [ "$what" != lm ]; then
  cd InfiniCore
  step xmake build -j 24; step xmake install; step xmake build -j 24 _infinicore; step xmake install _infinicore
  cd ..
fi
if [ "$what" != core ]; then
  cd InfiniLM
  step xmake build -j 24 _infinilm; step xmake install _infinilm
  cd ..
fi
echo REBUILD_OK
