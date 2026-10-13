#!/bin/bash
# Capture the official pipeline's first DiT inputs (diagnostic only) for single-step benchmarking.
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
export H3_TEXT_ENCODER=${H3_TEXT_ENCODER:-tp_all}
export H3_POINTWISE_NUMERICS=exact_ref H3_ADALN_NUMERICS=exact_ref H3_DETERMINISTIC_BLAS=1 H3_FUSED_ADALN=1
export H3_ROPE_NUMERICS=baseline H3_QWEN_FP32_ROPE=1 H3_QWEN_NINETOOTHED=1
mkdir -p logs/k100/dit-step
$PYTHON tools/capture_first_dit.py --output-dump logs/k100/dit-step/astronaut_step0.pt fl2va \
  --model-path /root/private_data/models/MiniMax-H3 \
  --text 'An astronaut in an orange space suit smiles and slowly waves at the camera.' \
  --first-frame samples/astronaut.png --gpus 0,1,2,3 --steps 50 --seed 42 --num-frames 124 \
  --output logs/k100/dit-step/unused.mp4
echo CAPTURE_EXIT $?
