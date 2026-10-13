source /opt/dtk/env.sh
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export H3_ROOT=$PWD
export PYTHON=$PWD/.venv/bin/python INFINI_ROOT=$PWD/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/ninetoothed/src:$PWD/tools:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export INFINICORE_HYGON_MHA=${INFINICORE_HYGON_MHA:-aten}
# HIP_ENABLE_DEFERRED_LOADING=0 fails: DAS torch fatbin has an unresolvable FP8 symbol
