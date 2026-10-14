#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-quick}
case "$MODE" in quick|full|official|all) ;; *) echo 'Usage: reproduce.sh [quick|full|official|all]' >&2; exit 2;; esac
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
(cd "$H3_TREE"; shasum -a 256 -c SHA256SUMS >/dev/null)
bash "$H3_TREE/scripts/bootstrap.sh"
mkdir -p "$H3_TREE/work"
if [ "$MODE" = quick ]; then
 python "$H3_TREE/tools/bench_gfx906_pointwise.py" --seq 17 --dtype bf16 --seed 0 --repeats 3 --samples 1 --caller hip --aot-dir "$H3_TREE/work/aot" --json "$H3_TREE/work/quick.json"
fi
if [ "$MODE" = full ] || [ "$MODE" = all ]; then
 python "$H3_TREE/scripts/run_suite.py" --output "$H3_TREE/work/results"
fi
if [ "$MODE" = official ] || [ "$MODE" = all ]; then
 bash "$H3_TREE/scripts/build_infinicore.sh"
fi
echo "Z100L_OPERATOR_SUITE_OK $MODE"
