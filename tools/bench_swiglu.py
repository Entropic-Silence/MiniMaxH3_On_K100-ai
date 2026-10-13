#!/usr/bin/env python3
"""Diagnostic: tile shapes / warps of the H3 SwiGLU NineToothed kernel on the
SP4 shape (up/gate halves of the [1, 5726, 28672] gate_up output): bitwise
check against the current configuration and timing. Never used by generation."""
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "ninetoothed/src"), str(Path(__file__).resolve().parent)]
from h3_pointwise import make_swiglu  # noqa: E402

T, F = 5726, 14336
dev = "cuda:1"
torch.manual_seed(0)
gate_up = (torch.randn(1, T, 2 * F, device=dev) * 2).to(torch.bfloat16)
up, gate = gate_up[..., :F], gate_up[..., F:]
build = Path(tempfile.mkdtemp(prefix="swiglu_aot_"))
arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]


def make(**options):
    out_dir = build / "_".join(f"{k}{v}" for k, v in options.items())
    out_dir.mkdir()
    return make_swiglu(caller="hip", output_dir=out_dir, archs=[arch], **options)


def run(kernel):
    out = torch.empty(1, T, F, device=dev, dtype=torch.bfloat16)
    with torch.cuda.device(dev):
        kernel(up, gate, out)
    return out


def bench(kernel):
    out = torch.empty(1, T, F, device=dev, dtype=torch.bfloat16)
    with torch.cuda.device(dev):
        for _ in range(3):
            kernel(up, gate, out)
        torch.cuda.synchronize(dev)
        start = time.perf_counter()
        for _ in range(50):
            kernel(up, gate, out)
        torch.cuda.synchronize(dev)
    return (time.perf_counter() - start) / 50


reference = make(rows=16, cols=256, num_warps=4)
expected = run(reference)
print(f"current rows16 cols256 w4: {bench(reference) * 1e3:.3f} ms", flush=True)
configs = [tuple(int(v) for v in c.split(",")) for c in sys.argv[1:]] or [
    (1, 1024, 4), (1, 2048, 8), (2, 1024, 4), (4, 512, 4), (4, 1024, 8), (8, 512, 8),
    (8, 256, 4), (16, 512, 8), (32, 128, 4), (2, 2048, 8), (1, 4096, 8)]
for rows, cols, warps in configs:
    try:
        kernel = make(rows=rows, cols=cols, num_warps=warps)
        same = torch.equal(run(kernel), expected)
        print(f"rows{rows} cols{cols} w{warps}: {bench(kernel) * 1e3:.3f} ms bitwise {same}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"rows{rows} cols{cols} w{warps}: failed {type(exc).__name__}: {str(exc)[:120]}", flush=True)
