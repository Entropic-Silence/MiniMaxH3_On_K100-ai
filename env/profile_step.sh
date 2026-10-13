#!/bin/bash
# hipprof kernel statistics for one SP4 DiT step on the captured input.
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
export H3_POINTWISE_NUMERICS=exact_ref H3_ADALN_NUMERICS=exact_ref H3_DETERMINISTIC_BLAS=1 H3_FUSED_ADALN=1 H3_ROPE_NUMERICS=baseline
export H3_PARALLEL=${H3_PARALLEL:-sp}
TAG=${1:-sp4}
mkdir -p logs/k100/prof/$TAG
cd logs/k100/prof/$TAG
/opt/dtk/bin/hipprof --hip-trace --stats -o $TAG $PYTHON $H3_ROOT/tools/bench_dit_step.py \
  --model-path /root/private_data/models/MiniMax-H3 --input-dump $H3_ROOT/logs/k100/dit-step/astronaut_step0.pt \
  --gpus 0,1,2,3 --warmup 1 --repeats 1 > prof.log 2>&1
echo PROF_EXIT $?
