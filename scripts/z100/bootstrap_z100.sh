#!/usr/bin/env bash
# One-time preparation of a fresh Z100 pod, in the order the obstacles appear.
# Run it inside the pod:  bash "$WORK_HOME/h3/scripts/z100/bootstrap_z100.sh"
#
# Why each step is here (all measured on the contest pod, 2026-10-13):
#   * the image has no xmake, pybind11 or triton, and no /opt/rocm;
#   * GitHub is unreliable from this network (HTTP/2 framing errors, TLS
#     resets, timeouts), so xmake's package repositories are unreachable and
#     the sources are uploaded over ssh instead of cloned;
#   * pybind11 3.x crashes this tree during pybind registration, so the build
#     needs 2.x -- and the whole C++ API must be compiled against the same
#     one, since both use the identical include path;
#   * upstream Triton has no gfx906 backend, so NineToothed kernels cannot be
#     generated here; the build therefore omits --h3-ninetoothed.
set -euo pipefail
: "${WORK_HOME:?enter the pod with WORK_HOME set, as the contest guide shows}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/setup_env_z100.sh"
export PATH=/root/.local/bin:$PATH

if ! command -v xmake >/dev/null 2>&1; then
  echo "== installing xmake 3.0.5"
  curl -fsSL -o /tmp/xmake.run \
    https://github.com/xmake-io/xmake/releases/download/v3.0.5/xmake-v3.0.5.gz.run
  sh /tmp/xmake.run
fi
# Without this xmake blocks trying to clone its package repositories.
xmake global --network=private

echo "== python build dependencies"
python -m pip install -q "pybind11<3"
python -c "import pybind11; print('pybind11', pybind11.__version__)"

echo "== environment"
python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("devices", torch.cuda.device_count(),
      torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
PY
echo "BOOTSTRAP_Z100_OK  (next: scripts/z100/build_z100.sh core)"
