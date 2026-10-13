#!/bin/bash
# P0.6: unmodified legacy reference on K100x4 (text GPUs 0,1; transformer GPUs 2,3 by default split)
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
TAG=${1:-legacy-astronaut-1}
H3_LOG_DIR=$PWD/logs/k100/$TAG bash scripts/run_case.sh fl2va --legacy \
  --model-path /root/private_data/models/MiniMax-H3 \
  --text 'An astronaut in an orange space suit smiles and slowly waves at the camera.' \
  --first-frame samples/astronaut.png --gpus 0,1,2,3 --legacy-memory-fraction 0.85 \
  --steps 50 --seed 42 --num-frames 124 --output $PWD/logs/k100/$TAG.mp4
echo "LEGACY_EXIT $?"
