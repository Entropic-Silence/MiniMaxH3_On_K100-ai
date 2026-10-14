#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
if ! command -v xmake >/dev/null; then
  echo 'Install xmake 3.0.5 in the POD before running official/all (quick/full do not require xmake).' >&2
  exit 1
fi
if [ ! -d "$INFINICORE_DIR" ]; then tar -xzf "$H3_TREE/vendor/InfiniCore-source.tar.gz" -C "$H3_TREE"; fi
cd "$INFINICORE_DIR"
export H3_OFFLINE_BUILD=1
export H3_PYBIND11_INCLUDE=$(python -c 'import pybind11; print(pybind11.get_include())')
export H3_PYTHON_INCLUDE=$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')
xmake f -y -cv --hygon-dcu=y --hygon-arch="$HYGON_ARCH" --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y --omp=n
xmake build -j "${BUILD_JOBS:-16}"
xmake install
xmake build -j "${BUILD_JOBS:-16}" _infinicore
xmake install _infinicore
python -m pip install -e . --no-deps --no-build-isolation
python test/infinicore/ops/timestep_embedding.py --hygon
