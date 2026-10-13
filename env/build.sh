#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/../.."
source /opt/dtk/env.sh
[ -f .venv/pyvenv.cfg ] || python3 -m venv --system-site-packages --without-pip .venv
.venv/bin/python -m pip install --no-index --no-deps --find-links third_party/wheels \
  "scikit-image==0.24.0" "av==14.2.0" "imageio==2.37.0" "imageio-ffmpeg==0.6.0" \
  "soundfile==0.13.1" "lazy-loader==0.4" "tifffile==2025.5.10" "pywavelets==1.8.0" "janus==2.0.0"
H3_VENV=$PWD/.venv bash scripts/setup_runtime.sh
export PYTHON=$PWD/.venv/bin/python INFINI_ROOT=$PWD/.infini H3_BUILD_JOBS=24
bash scripts/build_all.sh
echo BUILD_ALL_DONE
