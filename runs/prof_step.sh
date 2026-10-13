#!/bin/bash
# Paths are overridable: H3_ROOT (the source tree), H3_WORK (scratch: runs/bench/prof), H3_MODEL (the model).
# usage: prof_step.sh TAG  -- hipprof trace of one DiT step (captured astronaut input), current defaults
TAG=$1; shift
ROOT=${H3_ROOT:-/root/h3-work/h3-extreme}
source /opt/dtk/env.sh >/dev/null 2>&1
export INFINI_ROOT=$ROOT/.infini LD_LIBRARY_PATH=$ROOT/.infini/lib:$LD_LIBRARY_PATH
mkdir -p ${H3_WORK:-/root/h3-work}/prof/$TAG && cd ${H3_WORK:-/root/h3-work}/prof/$TAG
/opt/dtk/bin/hipprof --hip-trace --stats -o $TAG $ROOT/.venv/bin/python $ROOT/tools/bench_dit_step.py \
  --model-path ${H3_MODEL:-/root/h3-work/models/MiniMax-H3} --input-dump $ROOT/logs/k100/dit-step/astronaut_step0.pt \
  --gpus 0,1,2,3 --warmup 1 --repeats 1 "$@" > prof.log 2>&1
echo PROF_EXIT $?
ls
for d in 0 1 2 3; do echo "== device $d"; $ROOT/.venv/bin/python $ROOT/tools/prof_timeline.py $(ls *.db | head -1) $d; done
