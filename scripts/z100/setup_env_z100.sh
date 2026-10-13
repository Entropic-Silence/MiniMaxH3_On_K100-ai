#!/usr/bin/env bash
# Environment for the contest's Hygon Z100 pod, as the organizers' guide
# prescribes (DTK, xmake, rocm_smi library paths). Source it in every pod
# shell:  source "$WORK_HOME/h3/scripts/z100/setup_env_z100.sh"
source /opt/dtk/env.sh

export XMAKE_ROOT=y
export WORK_HOME="${WORK_HOME:?enter the pod with WORK_HOME set (see the guide)}"
export INFINI_ROOT=${INFINI_ROOT:-/root/.infini}
export PATH="/root/.local/bin:/opt/dtk/cuda/cuda/bin:${PATH}"
# librocm_smi64.so.2 lives in the .hyhal paths; without them importing torch fails.
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"

# This tree (the K100 work) and the organizers' clones.
export H3_TREE=${H3_TREE:-$WORK_HOME/h3}
# This tree's InfiniCore (uploaded with ops/zpush_core.sh); the organizers'
# clone lives in $WORK_HOME/Projects/InfiniCore if you prefer that one.
export INFINICORE_DIR=${INFINICORE_DIR:-$H3_TREE/InfiniCore}
export INFINILM_DIR=${INFINILM_DIR:-$WORK_HOME/Projects/InfiniLM}
export PYTHONPATH="$H3_TREE/ninetoothed/src:$H3_TREE/tools:${PYTHONPATH:-}"
export HYGON_ARCH=${HYGON_ARCH:-gfx906}
