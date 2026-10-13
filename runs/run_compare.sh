#!/bin/bash
# Paths are overridable: H3_ROOT (the source tree), H3_WORK (scratch: runs/bench/prof), H3_MODEL (the model).
# usage: run_compare.sh TAG [extra minimax_h3_compare.py args]
# The organizers' comparison script (legacy PyTorch vs InfiniLM, SSIM + timings) for FL2VA, with an
# evaluator-like environment (library paths only, offline). Results in ${H3_WORK:-/root/h3-work}/compare/TAG/.
TAG=$1; shift
ulimit -c 0
ROOT=${H3_ROOT:-/root/h3-work/h3-extreme}
source /opt/dtk/env.sh >/dev/null 2>&1
export INFINI_ROOT=$ROOT/.infini LD_LIBRARY_PATH=$ROOT/.infini/lib:$LD_LIBRARY_PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if ps -eo cmd | grep -qE "[e]xamples/minimax_h3.py|[t]race_pipeline.py"; then echo "BUSY: another minimax_h3 run is alive"; echo "EXIT 99"; exit 99; fi
mkdir -p ${H3_WORK:-/root/h3-work}/compare
cd $ROOT
$ROOT/.venv/bin/python $ROOT/InfiniLM/examples/minimax_h3_compare.py --tasks fl2va \
  --model-path ${MODEL:-${H3_MODEL:-/root/h3-work/models/MiniMax-H3}} --first-frame ${IMAGE:-$ROOT/samples/astronaut.png} \
  --output-dir ${H3_WORK:-/root/h3-work}/compare/$TAG "$@"
echo "EXIT $?"
