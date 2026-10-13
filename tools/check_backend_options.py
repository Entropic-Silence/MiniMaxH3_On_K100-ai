#!/usr/bin/env python3
"""Prove compiler-option propagation changes same-named generated HIP assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from h3_attention import make_attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    q, k, v = (torch.randn(1, 14, 8192, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    rows = []
    for name, options in [("baseline", {}), ("waves2", {"waves_per_eu": 2})]:
        outdir = args.output / name
        kernel = make_attention(caller="hip", output_dir=outdir, backend_options=options,
                                block_m=64, block_n=64, num_warps=4, num_stages=1)
        out = torch.empty_like(q)
        elapsed = []
        for repeat in range(2):
            kernel(q, k, v, 128**-0.5, out)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                kernel(q, k, v, 128**-0.5, out)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end) / 5)
        manifest = json.loads((outdir / "h3_full_attention_aligned.manifest.json").read_text())
        assembly = next(outdir.glob("h3_full_attention_aligned.*.amdgcn")).read_bytes()
        delta = (out.float() - ref.float()).abs()
        row = {"name": name, "options": options, "compiled_options": manifest["targets"][0]["options"],
               "assembly_sha256": hashlib.sha256(assembly).hexdigest(), "ms": elapsed,
               "max_error": delta.max().item(), "mean_error": delta.mean().item()}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"rows": rows, "assembly_differs": rows[0]["assembly_sha256"] != rows[1]["assembly_sha256"]}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["assembly_differs"]:
        raise SystemExit("Options did not change assembly")


if __name__ == "__main__":
    main()
