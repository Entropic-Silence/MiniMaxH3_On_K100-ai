#!/bin/bash
# Paths are overridable: H3_ROOT (the source tree), H3_WORK (scratch: runs/bench/prof), H3_MODEL (the model).
# usage: [ENV=..] bench_step.sh TAG [args]  -- repeated DiT step on captured astronaut input
TAG=$1; shift
ulimit -c 0
ROOT=${H3_ROOT:-/root/h3-work/h3-extreme}
source /opt/dtk/env.sh >/dev/null 2>&1
export INFINI_ROOT=$ROOT/.infini LD_LIBRARY_PATH=$ROOT/.infini/lib:$LD_LIBRARY_PATH
mkdir -p ${H3_WORK:-/root/h3-work}/bench
cd ${H3_WORK:-/root/h3-work}/bench
$ROOT/.venv/bin/python $ROOT/tools/bench_dit_step.py --model-path ${H3_MODEL:-/root/h3-work/models/MiniMax-H3} \
  --input-dump $ROOT/logs/k100/dit-step/astronaut_step0.pt --gpus 0,1,2,3 --warmup 2 --repeats 4 \
  --report ${H3_WORK:-/root/h3-work}/bench/$TAG.json "$@" > ${H3_WORK:-/root/h3-work}/bench/$TAG.log 2>&1
echo "BENCH_EXIT $?"
python3 -c "import json;d=json.load(open('${H3_WORK:-/root/h3-work}/bench/$TAG.json'));print({k:v for k,v in d.items() if k not in ('outputs',)})" 2>/dev/null | cut -c1-600
