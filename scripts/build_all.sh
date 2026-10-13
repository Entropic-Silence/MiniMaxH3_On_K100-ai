#!/usr/bin/env bash
set -euo pipefail
H3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
H3_PYTHON=${PYTHON:-python}
H3_KERNELS_ONLY=0
H3_CONFIG=()
while (($#)); do
  case "$1" in
    --kernels-only) H3_KERNELS_ONLY=1; shift ;;
    --config) H3_CONFIG=(--config "$2"); shift 2 ;;
    *) echo "Usage: $0 [--kernels-only] [--config attention-build.json]" >&2; exit 2 ;;
  esac
done
set +u
source "${DTK_ENV:-/opt/dtk/env.sh}"
set -u
export XMAKE_ROOT=y H3_OFFLINE_BUILD=1
export INFINI_ROOT=${INFINI_ROOT:-"$H3_ROOT/.infini"}
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$H3_ROOT/ninetoothed/src:$H3_ROOT/tools:${PYTHONPATH:-}"
export HYGON_CUDA_HOME=${HYGON_CUDA_HOME:-/opt/dtk/cuda/cuda}
HYGON_ARCH=$($H3_PYTHON -c 'from ninetoothed.platform import current_target; print(current_target().arch)')
export HYGON_ARCH
mkdir -p "$H3_ROOT/build" "$H3_ROOT/logs/build"
"$H3_PYTHON" "$H3_ROOT/tools/detect_hw.py" --output "$H3_ROOT/logs/build/hardware.json"
# Never resolve or upgrade accelerator packages. Install only from the supplied wheels.
"$H3_PYTHON" -m pip install --no-index --no-deps --find-links "$H3_ROOT/third_party/wheels" \
  'hatchling==1.27.0' 'pathspec==0.12.1' 'trove-classifiers==2025.9.11.17' 'pybind11==2.13.6'
"$H3_PYTHON" -m pip wheel --no-deps --no-build-isolation --no-index \
  "$H3_ROOT/ninetoothed" -w "$H3_ROOT/third_party/wheels"
"$H3_PYTHON" "$H3_ROOT/scripts/build_h3_attention.py" "${H3_CONFIG[@]}"
"$H3_PYTHON" -m pytest -q "$H3_ROOT/tests/test_aot_hip.py" "$H3_ROOT/tests/test_h3_pointwise.py"
if ((H3_KERNELS_ONLY)); then exit 0; fi

"$H3_PYTHON" -m pip install --no-index --no-deps --find-links "$H3_ROOT/third_party/wheels" \
  'scikit-image==0.24.0' 'av==14.2.0' 'imageio==2.37.0' 'imageio-ffmpeg==0.6.0' \
  'soundfile==0.13.1' 'lazy-loader==0.4' 'tifffile==2025.5.10' 'pywavelets==1.8.0' 'janus==2.0.0'
# Use the approved isolated runtime; never resolve protected accelerator packages here.
"$H3_PYTHON" "$H3_ROOT/scripts/check_runtime.py"

if ! command -v xmake >/dev/null; then
  H3_XMAKE_SRC="$H3_ROOT/build/xmake-src"
  if [[ ! -f "$H3_XMAKE_SRC/configure" ]]; then
    sh "$H3_ROOT/third_party/xmake-v3.1.1.gz.run" --noexec --target "$H3_XMAKE_SRC" --noprogress
  fi
  (cd "$H3_XMAKE_SRC" && ./configure --prefix="$H3_ROOT/.local" && make -j8 && make install)
  export PATH="$H3_ROOT/.local/bin:$PATH"
fi
export H3_PYBIND11_INCLUDE=$($H3_PYTHON -c 'import pybind11; print(pybind11.get_include())')
export H3_PYTHON_INCLUDE=$($H3_PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
export INFINILM_CXX11_ABI=$($H3_PYTHON -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
H3_FLASH_SITE=$($H3_PYTHON -c 'import flash_attn,pathlib; print(pathlib.Path(flash_attn.__file__).parents[1])')
H3_BUILD_JOBS=${H3_BUILD_JOBS:-8}
(
  cd "$H3_ROOT/InfiniCore"
  # OpenMP's xmake package resolver requires network; CPU work is small in this workload.
  xmake f -y --cuda="$HYGON_CUDA_HOME" --hygon-dcu=y --hygon-arch="$HYGON_ARCH" \
    --h3-ninetoothed=y --ccl=y --aten=y --omp=n --flash-attn="$H3_FLASH_SITE"
  xmake build -j "$H3_BUILD_JOBS"
  xmake install
  xmake build -j "$H3_BUILD_JOBS" _infinicore
  xmake install _infinicore
  "$H3_PYTHON" -m pip install --no-deps --no-build-isolation --no-index -e .
)
(
  cd "$H3_ROOT/InfiniLM"
  xmake f -y --cxx11-abi="$INFINILM_CXX11_ABI"
  xmake build -j "$H3_BUILD_JOBS" _infinilm
  xmake install _infinilm
  "$H3_PYTHON" -m pip install --no-deps --no-build-isolation --no-index -e .
)
"$H3_PYTHON" -c 'import infinicore; import infinilm; from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner'
"$H3_PYTHON" -c 'from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline'
