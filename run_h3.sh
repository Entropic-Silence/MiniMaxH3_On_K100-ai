#!/bin/bash
# One-command MiniMax-H3 generation on K100-AI x4 with every optimization of
# this image enabled. Arguments go unchanged to the official CLI
# (InfiniLM/examples/minimax_h3.py); relative paths are resolved from the
# directory you call this script from. Example:
#
#   ./run_h3.sh fl2va --model-path /root/h3-work/models/MiniMax-H3 \
#       --text "An astronaut in an orange space suit smiles and waves." \
#       --first-frame samples/astronaut.png --gpus 0,1,2,3 \
#       --steps 50 --seed 42 --num-frames 124 --output out.mp4
#
# Any variable below can be overridden from the environment (see README).
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${DTK_ENV:-/opt/dtk/env.sh}"
export INFINI_ROOT=${INFINI_ROOT:-$ROOT/.infini}
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ninetoothed/src:$ROOT/tools:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Kernels and parallelism (all on by default in this image).
export INFINICORE_HYGON_MHA=${INFINICORE_HYGON_MHA:-aten}      # vendor flash attention
# H3_PARALLEL unset: Ulysses sequence parallel DiT on >= 4 cards, tensor parallel below.
# Numerics of the validated configuration (reference BF16 rounding).
export H3_POINTWISE_NUMERICS=${H3_POINTWISE_NUMERICS:-exact_ref} H3_ADALN_NUMERICS=${H3_ADALN_NUMERICS:-exact_ref}
export H3_DETERMINISTIC_BLAS=${H3_DETERMINISTIC_BLAS:-1} H3_FUSED_ADALN=${H3_FUSED_ADALN:-1}
export H3_ROPE_NUMERICS=${H3_ROPE_NUMERICS:-baseline}
export H3_QWEN_FP32_ROPE=${H3_QWEN_FP32_ROPE:-1} H3_QWEN_NINETOOTHED=${H3_QWEN_NINETOOTHED:-1}

exec "${PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/InfiniLM/examples/minimax_h3.py" "$@"
