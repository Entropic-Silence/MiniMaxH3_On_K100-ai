#!/usr/bin/env bash
# Build InfiniCore (and optionally InfiniLM) for the Z100 pod, generate this
# tree's NineToothed AOT kernels for its architecture, and run the operator
# tests. Usage, inside the pod:
#   bash "$WORK_HOME/h3/scripts/z100/build_z100.sh" [core|kernels|lm|all]
set -euo pipefail
WHAT=${1:-all}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_env_z100.sh"
git config --global --add safe.directory "$INFINICORE_DIR" || true
git config --global --add safe.directory "$INFINILM_DIR" || true

if [ "$WHAT" = core ] || [ "$WHAT" = all ]; then
  cd "$INFINICORE_DIR"
  # Offline build: this network cannot reach xmake's package repositories, so
  # pybind11 comes from the installed wheel (2.x -- see bootstrap_z100.sh).
  export H3_OFFLINE_BUILD=1
  export H3_PYBIND11_INCLUDE=$(python -c 'import pybind11; print(pybind11.get_include())')
  export H3_PYTHON_INCLUDE=$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')
  xmake f -y -cv --hygon-dcu=y --hygon-arch="$HYGON_ARCH" \
      --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y --omp=n
  xmake build -j 16
  xmake install
  xmake build -j 16 _infinicore
  xmake install _infinicore
  python -m pip install -e . --no-deps --no-build-isolation
  python -c "import torch; import infinicore; print('infinicore', infinicore.__file__)"
  python test/infinicore/ops/timestep_embedding.py --hygon || true
fi

if [ "$WHAT" = kernels ]; then
  # The H3 NineToothed kernels of this tree, AOT for the pod's architecture.
  # Not part of "all" on gfx906: upstream Triton has no backend for it
  # ("error: unsupported target: 'gfx906'"), so this needs a Hygon DAS triton.
  # tools/probe_triton_arch.py qualifies a candidate before trying this.
  cd "$H3_TREE"
  python scripts/build_h3_attention.py --archs "$HYGON_ARCH" \
      --output "$INFINICORE_DIR/build/h3_ninetoothed"
fi

if [ "$WHAT" = lm ] || [ "$WHAT" = all ]; then
  cd "$INFINILM_DIR"
  xmake build -j 16 infinicore_infer
  xmake install infinicore_infer
  xmake build -j 16 _infinilm
  xmake install _infinilm
  python -m pip install -e . --no-deps --no-build-isolation
fi
echo "BUILD_Z100_OK $WHAT"
