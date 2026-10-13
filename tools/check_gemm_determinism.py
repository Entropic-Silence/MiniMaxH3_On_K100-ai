#!/usr/bin/env python3
"""Diagnostic: is InfiniCore's linear (and PyTorch's mm) bitwise repeatable on
the H3 SP shapes? Runs each GEMM several times on one device and reports
whether the outputs match the first run. Never used by generation."""
import os

import torch

os.environ.setdefault("H3_DETERMINISTIC_BLAS", "1")
import infinicore  # noqa: E402
import infinicore.nn.functional as F  # noqa: E402

SHAPES = {  # name: (M, N, K, dtype)
    "qkv_group": (5726, 10752, 5376, torch.bfloat16),
    "out": (5726, 5376, 7168, torch.bfloat16),
    "gate_up": (5726, 28672, 5376, torch.bfloat16),
    "down": (5726, 5376, 14336, torch.bfloat16),
    "adaln": (3, 24192, 2688, torch.bfloat16),
    "time_1": (2, 5376, 256, torch.float32),
    "time_2": (2, 2688, 5376, torch.float32),
    "refiner_qkv": (599, 21504, 5376, torch.bfloat16),
}


def main():
    device = 1
    torch.cuda.set_device(device)
    infinicore.set_device(infinicore.device("cuda", device))
    torch.manual_seed(0)
    for name, (m, n, k, dtype) in SHAPES.items():
        a = torch.randn(m, k, device=f"cuda:{device}", dtype=dtype)
        w = torch.randn(n, k, device=f"cuda:{device}", dtype=dtype)
        ia, iw = infinicore.from_torch(a), infinicore.from_torch(w)
        ref_ic = ref_t = None
        same_ic = same_t = 0
        for i in range(6):
            out = torch.empty(m, n, device=f"cuda:{device}", dtype=dtype)
            F.linear(ia, iw, out=infinicore.from_torch(out))
            infinicore.sync_device()
            t = torch.mm(a, w.t())
            torch.cuda.synchronize(device)
            if i == 0:
                ref_ic, ref_t = out.clone(), t.clone()
            else:
                same_ic += int(torch.equal(out, ref_ic))
                same_t += int(torch.equal(t, ref_t))
        print(f"{name:12s} infinicore repeat-equal {same_ic}/5   torch repeat-equal {same_t}/5   "
              f"ic-vs-torch equal {torch.equal(ref_ic, ref_t)}", flush=True)


if __name__ == "__main__":
    main()
