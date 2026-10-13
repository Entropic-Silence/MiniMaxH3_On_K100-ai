#!/usr/bin/env python3
"""Diagnostic: repeat the H3 time embedder chain (timestep_embedding ->
F32 linear+bias -> silu -> F32 linear+bias) with InfiniCore on one device and
check bitwise repeatability of each stage. Never used by generation."""
import os

import torch

os.environ.setdefault("H3_DETERMINISTIC_BLAS", "1")
import infinicore  # noqa: E402
import infinicore.nn.functional as F  # noqa: E402
from infinicore.ops.timestep_embedding import timestep_embedding  # noqa: E402


def to_t(x):
    out = torch.empty(tuple(x.shape), dtype=infinicore.utils.to_torch_dtype(x.dtype), device="cuda:1")
    infinicore.from_torch(out).copy_(x)
    infinicore.sync_device()
    return out


def main():
    torch.cuda.set_device(1)
    infinicore.set_device(infinicore.device("cuda", 1))
    torch.manual_seed(0)
    t = torch.tensor([0.37, 0.91], device="cuda:1", dtype=torch.float32)
    w1 = torch.randn(5376, 256, device="cuda:1") * 0.05
    b1 = torch.randn(5376, device="cuda:1") * 0.05
    w2 = torch.randn(2688, 5376, device="cuda:1") * 0.01
    b2 = torch.randn(2688, device="cuda:1") * 0.01
    it, iw1, ib1, iw2, ib2 = (infinicore.from_torch(x) for x in (t, w1, b1, w2, b2))
    refs = None
    same = [0, 0, 0, 0]
    for i in range(8):
        e = timestep_embedding(it, 256)
        h = F.linear(e, iw1, ib1)
        s = F.silu(h)
        o = F.linear(s, iw2, ib2)
        stages = [to_t(x) for x in (e, h, s, o)]
        if refs is None:
            refs = stages
        else:
            for j in range(4):
                same[j] += int(torch.equal(stages[j], refs[j]))
    print("repeat-equal (of 7): embedding %d, linear_1 %d, silu %d, linear_2 %d" % tuple(same))


if __name__ == "__main__":
    main()
