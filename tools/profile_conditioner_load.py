#!/usr/bin/env python3
"""Diagnostic: where the Qwen3-VL condition encoder construction spends time."""
import argparse
import cProfile
import pstats
import time
from pathlib import Path

import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    MiniMaxH3ConditionEncoder(args.model_path / "text_encoder", dtype=torch.bfloat16,
                              tp_device_ids=[int(d) for d in args.gpus.split(",")])
    profiler.disable()
    print(f"conditioner create+load {time.perf_counter() - start:.2f} s", flush=True)
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
