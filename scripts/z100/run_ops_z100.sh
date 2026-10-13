#!/usr/bin/env bash
# Operator correctness + performance table on the Z100 pod (BF16 and FP16),
# plus the attention tiling sweep. Results land in $WORK_HOME/h3-z100/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_env_z100.sh"
OUT=${OUT:-$WORK_HOME/h3-z100}
SEQ=${SEQ:-8192}
HEADS=${HEADS:-14}
mkdir -p "$OUT"
cd "$H3_TREE"
for dtype in bf16 fp16; do
  echo "== operators, $dtype"
  python tools/bench_h3_ops.py --seq "$SEQ" --heads "$HEADS" --dtype "$dtype" \
      --json "$OUT/ops_$dtype.json" 2>&1 | tee "$OUT/ops_$dtype.log"
done
echo "== attention tiling sweep"
python tools/tune_attention.py --seq "$SEQ" --heads "$HEADS" \
    --output "$OUT/attention_tune.json" 2>&1 | tee "$OUT/attention_tune.log"
echo OPS_Z100_DONE
