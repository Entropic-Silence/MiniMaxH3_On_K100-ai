#!/usr/bin/env python3
"""Diagnostic: is a bf16 all-reduce of a buffer bitwise equal to all-reducing
its [0, cut) and [cut, n) parts separately? Finds the split granularity for
which RCCL's per-element summation order is preserved."""
import os
import sys

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    rows, width = 22901, 5376
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    base = (torch.randn(rows, width, device="cuda", generator=gen) * 4).to(torch.bfloat16)
    full = base.clone()
    dist.all_reduce(full)
    cuts_rows = [int(x) for x in sys.argv[1].split(",")]
    for cut in cuts_rows:
        parts = base.clone()
        dist.all_reduce(parts[:cut])
        dist.all_reduce(parts[cut:])
        same = torch.equal(full.view(torch.int16), parts.view(torch.int16))
        mismatch = (full.view(torch.int16) != parts.view(torch.int16)).nonzero()
        first = int(mismatch[0, 0]) if mismatch.numel() else -1
        flag = torch.tensor([0 if same else 1], device="cuda")
        dist.all_reduce(flag)
        if rank == 0:
            print(f"cut {cut} rows ({cut * width * 2 / 2**20:.2f} MiB): bitwise {'yes' if int(flag) == 0 else 'NO'}"
                  f" first mismatching row {first}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
