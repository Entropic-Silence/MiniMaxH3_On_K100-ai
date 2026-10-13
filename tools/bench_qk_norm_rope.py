#!/usr/bin/env python3
"""Diagnostic: the block-tiled q/k RMSNorm + RoPE NineToothed kernel against the
one-token-per-program kernel, on the SP4 layout (strided source view of the
fused QKV projection, permuted send buffer): bitwise check and timing."""
import sys
import time
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "ninetoothed/src"), str(Path(__file__).resolve().parent)]
from h3_pointwise import make_qk_norm_rope  # noqa: E402

T, P, HS, D = 5726, 4, 7, 128
torch.manual_seed(0)
dev = "cuda:1"
projected = torch.randn(T, 3, P * HS, D, device=dev, dtype=torch.bfloat16)
source = projected[:, 0]                        # [T, P*Hs, D], token stride 3*P*Hs*D
by_rank = source.as_strided((T, P, HS, D), (source.stride(0), HS * source.stride(1), source.stride(1), 1))
weight = (torch.rand(D, device=dev) + 0.5).to(torch.bfloat16)
cos = torch.rand(T, 48, device=dev).to(torch.bfloat16)
sin = torch.rand(T, 48, device=dev).to(torch.bfloat16)


def run(kernel):
    out = torch.empty(P, T, HS, D, device=dev, dtype=torch.bfloat16)
    y = out.permute(1, 0, 2, 3)
    with torch.cuda.device(dev):
        kernel(by_rank, by_rank, by_rank, weight, weight, weight, 1e-5, cos, sin, y, y, y)
    return out


def bench(kernel):
    for _ in range(3):
        run(kernel)
    torch.cuda.synchronize(dev)
    start = time.perf_counter()
    for _ in range(50):
        run(kernel)
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - start) / 50


import tempfile
build = Path(tempfile.mkdtemp(prefix="qk_aot_"))
arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]


def make(**options):
    out_dir = build / "_".join(f"{k}{v}" for k, v in options.items())
    out_dir.mkdir()
    return make_qk_norm_rope(caller="hip", output_dir=out_dir, archs=[arch], **options)


reference = make(block_tokens=1, num_warps=1)
print("reference built", flush=True)
expected = run(reference)
print(f"current (1 token, 1 rank, 1 warp): {bench(reference) * 1e3:.3f} ms", flush=True)
configs = [tuple(int(v) for v in c.split(",")) for c in sys.argv[1:]] or [(1, 4, 8, 1), (1, 4, 8, 2), (1, 4, 8, 4)]
for bt, ranks, rows, nw in configs:
    try:
        kernel = make(block_tokens=bt, ranks=ranks, rows=rows, num_warps=nw)
        same = torch.equal(run(kernel), expected)
        print(f"bt{bt} ranks{ranks} rows{rows} w{nw}: {bench(kernel) * 1e3:.3f} ms bitwise {same}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"bt{bt} ranks{ranks} rows{rows} w{nw}: failed {type(exc).__name__}: {str(exc)[:160]}", flush=True)
