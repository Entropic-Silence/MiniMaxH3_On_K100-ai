#!/usr/bin/env python3
"""Diagnostic: run the native DiT N times on one captured input inside one
process and report which outputs are bitwise identical (and the relative L2
of each to the first). Never used by generation."""
import argparse
import hashlib
from pathlib import Path

import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--runs", type=int, default=4)
    args = parser.parse_args()
    devices = [int(d) for d in args.gpus.split(",")]
    captured = torch.load(args.input_dump, weights_only=True)
    inputs = {k: v.to(f"cuda:{devices[0]}") for k, v in captured["inputs"].items()}
    runner = MiniMaxH3TransformerRunner(args.model_path / "transformer", device="nvidia",
                                        tp_device_ids=devices)
    adapter = MiniMaxH3DiffusersAdapter(runner)
    outputs = []
    with torch.inference_mode():
        def input_digest():
            torch.cuda.synchronize(devices[0])
            return {k: hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]
                    for k, v in inputs.items()}
        base = input_digest()
        print("timestep", inputs["timestep"].tolist(), flush=True)
        for i in range(args.runs):
            out = adapter(**inputs)
            changed = [k for k, v in input_digest().items() if v != base[k]]
            if changed:
                print(f"run {i}: inputs changed by the forward: {changed}", flush=True)
            video = out.sample.float().cpu()
            digest = hashlib.sha256(out.sample.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
            rel = 0.0 if not outputs else float((video - outputs[0]).norm() / outputs[0].norm())
            outputs.append(video)
            print(f"run {i}: sha {digest} rel_l2_vs_run0 {rel:.6f}", flush=True)


if __name__ == "__main__":
    main()
