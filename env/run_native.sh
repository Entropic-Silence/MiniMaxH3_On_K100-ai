#!/bin/bash
# Native InfiniLM path on K100x4 with current best numerics switches (from BW100 checkpoint).
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
TAG=${1:-native-tp4-astronaut-1}
[ -n "${H3_TEXT_ENCODER:-}" ] && export H3_TEXT_ENCODER
export H3_POINTWISE_NUMERICS=${H3_POINTWISE_NUMERICS:-exact_ref} H3_ADALN_NUMERICS=${H3_ADALN_NUMERICS:-exact_ref}
export H3_DETERMINISTIC_BLAS=${H3_DETERMINISTIC_BLAS:-1} H3_FUSED_ADALN=${H3_FUSED_ADALN:-1} H3_ROPE_NUMERICS=${H3_ROPE_NUMERICS:-baseline}
export H3_QWEN_FP32_ROPE=${H3_QWEN_FP32_ROPE:-1} H3_QWEN_NINETOOTHED=${H3_QWEN_NINETOOTHED:-1}
GPUS=${GPUS:-0,1,2,3}
H3_LOG_DIR=$PWD/logs/k100/$TAG bash scripts/run_case.sh fl2va \
  --model-path ${MODEL:-/root/private_data/models/MiniMax-H3} \
  --text 'An astronaut in an orange space suit smiles and slowly waves at the camera.' \
  --first-frame samples/astronaut.png --gpus $GPUS \
  --steps 50 --seed 42 --num-frames 124 --output $PWD/logs/k100/$TAG.mp4
echo "NATIVE_EXIT $?"
