#!/usr/bin/env python3
"""Measure H3 NineToothed full attention; independent reference only in this tool."""
import argparse
import itertools
import json
from pathlib import Path
import time
import torch
from h3_attention import make_attention


def elapsed(fn, repeats):
    fn()
    torch.cuda.synchronize()
    results = []
    for _ in range(2):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        results.append(start.elapsed_time(end) / repeats / 1000)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=16384)
    p.add_argument("--heads", type=int, default=14)
    p.add_argument("--configs", help="BM,BN,warps,stages;...")
    p.add_argument("--backend-options", default="{}")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--caller", choices=["torch", "hip"], default="torch")
    p.add_argument("--strided", action="store_true")
    p.add_argument("--numerics", choices=["fp32_qk", "bf16_prescale", "fp32_qk_dotacc", "fp32_qk_split"], default="fp32_qk")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configs = [tuple(map(int, c.split(","))) for c in args.configs.split(";")] if args.configs else list(
        itertools.product([64, 128], [32, 64, 128], [4, 8], [1, 2]))
    torch.manual_seed(2026)
    if args.strided:
        packed = torch.randn(1, args.seq, 3, args.heads, 128, device="cuda", dtype=torch.bfloat16)
        q, k, v = (packed[:, :, i].transpose(1, 2) for i in range(3))
    else:
        q, k, v = [torch.randn(1, args.heads, args.seq, 128, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)]
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # Check selected queries against FP32 math without materializing a full S*S matrix.
    selected = torch.linspace(0, args.seq-1, min(16, args.seq), device="cuda").long()
    fp_logits = q[:, :, selected].float() @ k.float().transpose(-2, -1) * (128**-0.5)
    fp_ref = fp_logits.softmax(-1) @ v.float()
    del fp_logits
    out = torch.empty_like(q)
    props = torch.cuda.get_device_properties(0)
    records = []
    with args.output.open("w") as log:
        for i, (bm, bn, w, s) in enumerate(configs):
            row = {"seq": args.seq, "heads": args.heads, "block_m": bm, "block_n": bn,
                   "num_warps": w, "num_stages": s, "backend_options": json.loads(args.backend_options),
                   "arch": props.gcnArchName, "device": props.name, "torch": torch.__version__,
                   "hip": torch.version.hip, "caller": args.caller, "strided": args.strided,
                   "numerics": args.numerics}
            try:
                kernel = make_attention(bm, bn, w, s, backend_options=row["backend_options"],
                    caller=args.caller, output_dir=args.output.parent / f"aot_{i}",
                    kernel_name=f"h3_attention_{i}", numerics=args.numerics)
                times = elapsed(lambda: kernel(q, k, v, 128**-0.5, out), 3)
                delta = (out.float() - ref.float()).abs()
                fp_delta = (out[:, :, selected].float() - fp_ref).abs()
                row.update(seconds=times, tflops=4 * args.seq**2 * args.heads * 128 / times[1] / 1e12,
                    max_error=delta.max().item(), mean_error=delta.mean().item(),
                    fp32_sample_max_error=fp_delta.max().item(), fp32_sample_mean_error=fp_delta.mean().item())
                row["sdpa_alignment_passed"] = row["max_error"] <= 0.002 and row["mean_error"] <= 0.0002
                row["fp32_sample_passed"] = row["fp32_sample_max_error"] <= 0.002 and row["fp32_sample_mean_error"] <= 0.0002
                row["passed"] = row["sdpa_alignment_passed"] and row["fp32_sample_passed"]
            except Exception as exc:
                row.update(passed=False, error=repr(exc))
            records.append(row)
            summary = dict(row)
            if "error" in summary:
                summary["error"] = summary["error"][-400:]
            print(json.dumps(summary), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        ref_times = elapsed(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), 3)
        row = {"reference": "torch_sdpa", "seconds": ref_times,
               "tflops": 4 * args.seq**2 * args.heads * 128 / ref_times[1] / 1e12}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
    valid = [r for r in records if r["passed"]]
    if not valid:
        raise SystemExit("No configuration passed correctness")
    best = min(valid, key=lambda r: r["seconds"][1])
    args.output.with_suffix(".best.json").write_text(json.dumps(best, indent=2) + "\n")


if __name__ == "__main__":
    main()
