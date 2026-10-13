#!/bin/bash
# Incremental native rebuild on K100 (same flags as scripts/build_all.sh).
set -eo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
export XMAKE_ROOT=y H3_OFFLINE_BUILD=1 PATH=$PWD/.local/bin:$PATH
export HYGON_CUDA_HOME=/opt/dtk/cuda/cuda
export H3_PYBIND11_INCLUDE=$($PYTHON -c 'import pybind11; print(pybind11.get_include())')
export H3_PYTHON_INCLUDE=$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
export INFINILM_CXX11_ABI=$($PYTHON -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
export HYGON_ARCH=$($PYTHON -c 'from ninetoothed.platform import current_target; print(current_target().arch)')
if [ "$1" = "--kernels" ]; then
  $PYTHON scripts/build_h3_attention.py
fi
(cd InfiniCore && xmake build -j 24 && xmake install && xmake build -j 24 _infinicore && xmake install _infinicore)
(cd InfiniLM && xmake build -j 24 _infinilm && xmake install _infinilm)
$PYTHON -c 'import infinicore, infinilm; from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner; print("IMPORT_OK")'
echo REBUILD_DONE
