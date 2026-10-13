#!/usr/bin/env python3
"""Diagnostic: where the resident-mode DiT load spends its time once the
shards are already in pinned host memory (as after the entry's prefetch)."""
import argparse
import cProfile
import pstats
import time
from pathlib import Path

from infinilm.modeling_utils import prefetch_checkpoint_raw, _RAW_PREFETCH
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    path = args.model_path / "transformer"
    start = time.perf_counter()
    prefetch_checkpoint_raw(path, pinned=True)
    for future in list(_RAW_PREFETCH.values()):
        future.result()
    print(f"read+pin {time.perf_counter() - start:.1f} s", flush=True)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    MiniMaxH3TransformerRunner(path, device="nvidia",
                               tp_device_ids=[int(d) for d in args.gpus.split(",")])
    profiler.disable()
    print(f"runner create+load {time.perf_counter() - start:.2f} s", flush=True)
    pstats.Stats(profiler).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
