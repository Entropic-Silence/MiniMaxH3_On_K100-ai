#!/bin/bash
# Usage: run_compare_steps.sh TAG  (mode via H3_PARALLEL / H3_TP_OVERLAP env)
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
export H3_POINTWISE_NUMERICS=exact_ref H3_ADALN_NUMERICS=exact_ref H3_DETERMINISTIC_BLAS=1 H3_FUSED_ADALN=1 H3_ROPE_NUMERICS=baseline
TAG=$1
$PYTHON tools/compare_steps.py --model-path ${MODEL:-/root/private_data/models/MiniMax-H3} \
  --steps-dir logs/k100/legacy-steps/cat --report logs/k100/legacy-steps/cat-$TAG.json \
  --save-outputs logs/k100/legacy-steps/cat-$TAG-out.pt
echo COMPARE_EXIT $?
