#!/usr/bin/env python3
"""Diagnostic benchmark: repeat the native DiT on one captured real input.

Inputs come from ``capture_first_dit.py`` (the official pipeline's first DiT
call). Weight loading is excluded; every timed call runs the complete 50-block
forward with host synchronization on all ranks. Optionally saves the output or
compares it against a saved output (bitwise and relative L2) so a performance
change can be checked for numerical identity. Never used by generation.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner


def sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    devices = [int(d) for d in args.gpus.split(",")]
    captured = torch.load(args.input_dump, weights_only=True)
    assert captured["diagnostic_only"]
    inputs = {k: v.to(f"cuda:{devices[0]}") for k, v in captured["inputs"].items()}
    started = time.perf_counter()
    runner = MiniMaxH3TransformerRunner(args.model_path / "transformer", device="nvidia",
                                        tp_device_ids=devices)
    adapter = MiniMaxH3DiffusersAdapter(runner)
    load_seconds = time.perf_counter() - started
    times = []
    with torch.inference_mode():
        for i in range(args.warmup + args.repeats):
            sync(devices)
            t0 = time.perf_counter()
            output = adapter(**inputs)
            sync(devices)
            if i >= args.warmup:
                times.append(time.perf_counter() - t0)
    video, audio = output.sample.float().cpu(), output.audio_sample.float().cpu()
    report = {"diagnostic_only": True, "sequence_length": int(inputs["position_ids"].shape[0]),
              "devices": devices, "load_seconds": load_seconds, "seconds": times,
              "median_seconds": statistics.median(times)}
    if args.save_output:
        torch.save({"video": video, "audio": audio}, args.save_output)
    if args.compare:
        ref = torch.load(args.compare, weights_only=True)
        for name, value in (("video", video), ("audio", audio)):
            diff = value - ref[name]
            report[f"{name}_bitwise_equal"] = bool(torch.equal(value, ref[name]))
            report[f"{name}_relative_l2"] = float(diff.norm() / ref[name].norm().clamp_min(1e-12))
            report[f"{name}_max_abs"] = float(diff.abs().max())
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.report:
        args.report.write_text(text + "\n")


if __name__ == "__main__":
    main()
