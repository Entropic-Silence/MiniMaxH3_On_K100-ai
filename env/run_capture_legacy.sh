#!/bin/bash
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
O=logs/k100/legacy-steps/cat
mkdir -p $O
$PYTHON tools/capture_legacy_steps.py --steps-to-save 0,6,12,18,24,30,36,42,48 --output-dir $O \
  fl2va --legacy --legacy-memory-fraction 0.85 --model-path ${MODEL:-/root/private_data/models/MiniMax-H3} \
  --text 'A tabby cat slowly turns its head and blinks at the camera.' --first-frame samples/eval/chelsea.png \
  --gpus 0,1,2,3 --steps 50 --seed 7 --num-frames 124 --height 768 --width 768 --output $O/unused.mp4
echo CAPTURE_EXIT $?
