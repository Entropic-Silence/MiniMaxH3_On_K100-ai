#!/bin/bash
# Usage: bench_step.sh TAG [extra bench args]; numerics env as in run_native.sh.
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
export H3_POINTWISE_NUMERICS=${H3_POINTWISE_NUMERICS:-exact_ref} H3_ADALN_NUMERICS=${H3_ADALN_NUMERICS:-exact_ref}
export H3_DETERMINISTIC_BLAS=${H3_DETERMINISTIC_BLAS:-1} H3_FUSED_ADALN=${H3_FUSED_ADALN:-1} H3_ROPE_NUMERICS=${H3_ROPE_NUMERICS:-baseline}
TAG=$1; shift
$PYTHON tools/bench_dit_step.py --model-path /root/private_data/models/MiniMax-H3 \
  --input-dump logs/k100/dit-step/astronaut_step0.pt --gpus ${GPUS:-0,1,2,3} \
  --report logs/k100/dit-step/$TAG.json "$@"
echo BENCH_EXIT $?
