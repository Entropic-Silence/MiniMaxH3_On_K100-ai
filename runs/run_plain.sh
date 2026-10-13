#!/bin/bash
# Paths are overridable: H3_ROOT (the source tree), H3_WORK (scratch: runs/bench/prof), H3_MODEL (the model).
# usage: run_plain.sh TAG [extra env assignments via env]; mimics the evaluator: only library paths set
TAG=$1; shift
ulimit -c 0
ROOT=${H3_ROOT:-/root/h3-work/h3-extreme}
source /opt/dtk/env.sh >/dev/null 2>&1
export INFINI_ROOT=$ROOT/.infini LD_LIBRARY_PATH=$ROOT/.infini/lib:$LD_LIBRARY_PATH
mkdir -p ${H3_WORK:-/root/h3-work}/runs
cd $ROOT
# refuse to start next to a leftover run (it would OOM or skew timings)
if ps -eo cmd | grep -qE "[e]xamples/minimax_h3.py|[t]race_pipeline.py"; then echo "BUSY: another minimax_h3 run is alive"; echo "EXIT 99"; exit 99; fi
( while true; do echo "$(date +%s) $(hy-smi --showmeminfo vram 2>/dev/null | grep 'Used Memory' | awk '{print $NF}' | paste -sd, )"; sleep 1; done ) > ${H3_WORK:-/root/h3-work}/runs/$TAG-smi.log 2>&1 &
SMI=$!
# TRACE=1: run through tools/trace_pipeline.py (hashes of every component call -> $TAG.trace.jsonl)
ENTRY=$ROOT/InfiniLM/examples/minimax_h3.py
[ -n "$TRACE" ] && ENTRY="$ROOT/tools/trace_pipeline.py ${H3_WORK:-/root/h3-work}/runs/$TAG.trace.jsonl"
${PYSPY:+py-spy record --nonblocking --format raw -r 25 -o ${H3_WORK:-/root/h3-work}/runs/$TAG.pyspy --} $ROOT/.venv/bin/python $ENTRY fl2va --model-path ${MODEL:-${H3_MODEL:-/root/h3-work/models/MiniMax-H3}} \
  --text "${TEXT:-An astronaut in an orange space suit smiles and slowly waves at the camera.}" \
  --first-frame ${IMAGE:-samples/astronaut.png} --gpus ${GPUS:-0,1,2,3} \
  --steps 50 --seed 42 --num-frames ${FRAMES:-124} --output ${H3_WORK:-/root/h3-work}/runs/$TAG.mp4 "$@"
echo "EXIT $?"
kill $SMI
