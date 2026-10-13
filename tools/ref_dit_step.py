#!/usr/bin/env python3
"""Diagnostic: run the unmodified Diffusers H3 DiT on one captured input and
compare saved native outputs against it (relative L2 / max abs).

The reference is the fixed Diffusers MiniMaxH3Transformer3DModel dispatched
over the given GPUs with the official entrypoint's output-device hooks. It is
only used for numerical comparison, never by generation.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel


def compare(actual, reference):
    diff = actual.float() - reference.float()
    return {"relative_l2": float(diff.norm() / reference.float().norm().clamp_min(1e-12)),
            "max_abs": float(diff.abs().max()), "mean_abs": float(diff.abs().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.reference_output.exists():
        devices = [int(d) for d in args.gpus.split(",")]
        captured = torch.load(args.input_dump, weights_only=True)
        assert captured["diagnostic_only"]
        inputs = {k: v.to(f"cuda:{devices[0]}") for k, v in captured["inputs"].items()}
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            str(args.model_path / "transformer"), dtype=torch.bfloat16, device_map="balanced",
            max_memory={d: "54GiB" for d in devices}, low_cpu_mem_usage=True, local_files_only=True)
        model.eval().requires_grad_(False)
        official = Path(__file__).resolve().parents[1] / "InfiniLM/examples/minimax_h3.py"
        spec = importlib.util.spec_from_file_location("official_minimax_h3_ref", official)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.install_legacy_output_device_hooks(model)
        with torch.inference_mode():
            video, audio = model(**inputs, return_dict=False)
        torch.save({"video": video.float().cpu(), "audio": audio.float().cpu()}, args.reference_output)
        del model
    reference = torch.load(args.reference_output, weights_only=True)
    report = {"diagnostic_only": True, "reference": str(args.reference_output), "candidates": {}}
    for path in args.candidates:
        candidate = torch.load(path, weights_only=True)
        report["candidates"][path] = {name: compare(candidate[name], reference[name])
                                      for name in ("video", "audio")}
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n")


if __name__ == "__main__":
    main()
