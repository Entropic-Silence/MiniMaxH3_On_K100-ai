#!/usr/bin/env python3
"""Diagnostic: teacher-forced per-step comparison of the native DiT with legacy.

Loads the native runner once (mode from H3_PARALLEL) and evaluates every
`step_XX.pt` from `capture_legacy_steps.py` on the legacy trajectory's own
inputs, reporting relative L2 / max abs error against the legacy outputs.
"""
import argparse
import json
from pathlib import Path

import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner


def compare(actual, reference):
    diff = actual.float() - reference.float()
    return {"relative_l2": float(diff.norm() / reference.float().norm().clamp_min(1e-12)),
            "max_abs": float(diff.abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--steps-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--save-outputs", type=Path)
    args = parser.parse_args()
    devices = [int(d) for d in args.gpus.split(",")]
    runner = MiniMaxH3TransformerRunner(args.model_path / "transformer", device="nvidia",
                                        tp_device_ids=devices)
    adapter = MiniMaxH3DiffusersAdapter(runner)
    report = {"diagnostic_only": True, "steps": {}}
    outputs = {}
    for path in sorted(args.steps_dir.glob("step_*.pt")):
        record = torch.load(path, weights_only=True)
        assert record["diagnostic_only"]
        inputs = {k: v.to(f"cuda:{devices[0]}") for k, v in record["inputs"].items()}
        with torch.inference_mode():
            output = adapter(**inputs)
        video, audio = output.sample.float().cpu(), output.audio_sample.float().cpu()
        outputs[record["step"]] = {"video": video, "audio": audio}
        report["steps"][record["step"]] = {"video": compare(video, record["reference"]["video"]),
                                           "audio": compare(audio, record["reference"]["audio"])}
        print(json.dumps({"step": record["step"], **report["steps"][record["step"]]}), flush=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.save_outputs:
        torch.save(outputs, args.save_outputs)


if __name__ == "__main__":
    main()
