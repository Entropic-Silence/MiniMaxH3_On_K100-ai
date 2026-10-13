#!/usr/bin/env python3
"""Diagnostic evaluation driver: load one backend once, generate several FL2VA cases.

Reuses the unchanged official entrypoint's functions (input preparation,
pipeline loading, generation call, media encoding) so outputs match what the
CLI would produce for each case; only the model loading is shared. Used to
compare legacy / InfiniLM variants on more cases, never as a timing result.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_entry():
    path = ROOT / "InfiniLM/examples/minimax_h3.py"
    spec = importlib.util.spec_from_file_location("h3_official_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSONL: id,text,first_frame,seed,num_frames[,steps]")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--legacy-memory-fraction", type=float, default=0.85)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    entry = load_entry()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    pipe = None
    summary = []
    for case in cases:
        ns = argparse.Namespace(
            task="fl2va", model_path=args.model_path, text=case["text"], gpus=args.gpus,
            steps=case.get("steps", 50), seed=case["seed"], height=case.get("height"),
            width=case.get("width"), legacy=args.legacy, legacy_text_gpus=None,
            legacy_transformer_gpus=None, legacy_memory_fraction=args.legacy_memory_fraction,
            first_frame=str(ROOT / case["first_frame"]), last_frame=None,
            num_frames=case.get("num_frames", 124),
            output=str(args.output_dir / f"{case['id']}.mp4"))
        devices, text_devices, transformer_devices = entry.validate_args(ns)
        timings = entry.TimingReport(text_devices + transformer_devices if ns.legacy else devices)
        paths, generation_kwargs, _ = entry.prepare_generation_inputs(ns, timings)
        if pipe is None:
            pipe, _, _ = entry.load_pipeline(
                paths, ns.task, devices, timings, legacy=ns.legacy,
                legacy_text_devices=text_devices, legacy_transformer_devices=transformer_devices,
                legacy_memory_fraction=ns.legacy_memory_fraction)
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipe(**generation_kwargs)
        for device in (text_devices + transformer_devices if ns.legacy else devices):
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        entry.encode_video(result["videos"][0], fps=entry.FPS, output_path=ns.output,
                           audio=result["audio"][0], audio_sample_rate=result["sampling_rate"])
        summary.append({"id": case["id"], "output": ns.output, "generation_seconds": seconds})
        print("[case] " + json.dumps(summary[-1]), flush=True)
    (args.output_dir / "cases.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
