#!/usr/bin/env python3
"""Diagnostic: run the native DiT twice on one captured input with
H3_TRACE_DIR pointing at two directories, then list the traced tensors in
execution order with whether the two runs agree bitwise. Never used by
generation."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    devices = [int(d) for d in args.gpus.split(",")]
    captured = torch.load(args.input_dump, weights_only=True)
    inputs = {k: v.to(f"cuda:{devices[0]}") for k, v in captured["inputs"].items()}
    runner = MiniMaxH3TransformerRunner(args.model_path / "transformer", device="nvidia",
                                        tp_device_ids=devices)
    adapter = MiniMaxH3DiffusersAdapter(runner)
    dirs = []
    with torch.inference_mode():
        adapter(**inputs)  # first forward (synchronous collectives) untraced
        for i in range(args.runs):
            d = args.out / f"run{i}"
            d.mkdir(parents=True, exist_ok=True)
            os.environ["H3_TRACE_DIR"] = str(d)
            os.environ.setdefault("H3_TRACE_TOKENS", "4096")
            adapter(**inputs)
            dirs.append(d)
    os.environ.pop("H3_TRACE_DIR", None)
    names = sorted((p.name for p in dirs[0].iterdir()), key=lambda n: (dirs[0] / n).stat().st_mtime)
    for name in names:
        ref = np.fromfile(dirs[0] / name, dtype=np.uint8)
        status = []
        for d in dirs[1:]:
            other = np.fromfile(d / name, dtype=np.uint8)
            status.append("same" if other.size == ref.size and np.array_equal(other, ref) else "DIFF")
        print(f"{name:40s} {' '.join(status)}", flush=True)


if __name__ == "__main__":
    main()
