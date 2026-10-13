#!/usr/bin/env python3
"""RCCL all-to-all vs GEMM interference, the counterpart of bench_p2p_sdma.py.

torchrun --standalone --nproc_per_node=4 tools/bench_rccl_interference.py
Same pattern: every rank sends a 31 MiB block to every peer, 12 times, while a
GEMM loop runs on another stream. Diagnostic only.
"""
import argparse
import datetime
import json
import os
import time

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm", default="5728,10752,5376")
    parser.add_argument("--block-mib", type=float, default=31)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    block = int(args.block_mib * (1 << 20)) // 2
    src = torch.empty(block * world, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    m, n, k = map(int, args.gemm.split(","))
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    comm = torch.cuda.Stream()
    gemm = torch.cuda.Stream()
    reps = 40

    def gemms(count):
        with torch.cuda.stream(gemm):
            for _ in range(count):
                torch.mm(a, b.t())

    def exchanges(count):
        with torch.cuda.stream(comm):
            for _ in range(count):
                dist.all_to_all_single(dst, src)

    exchanges(2)
    gemms(2)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    gemms(reps)
    torch.cuda.synchronize()
    alone = time.perf_counter() - start
    dist.barrier()
    start = time.perf_counter()
    exchanges(12)
    torch.cuda.synchronize()
    comm_alone = time.perf_counter() - start
    dist.barrier()
    start = time.perf_counter()
    gemms(reps)
    exchanges(12)
    torch.cuda.synchronize()
    both = time.perf_counter() - start
    row = {"rank": rank, "gemms_alone_s": alone, "copies_alone_s": comm_alone,
           "gemms_plus_copies_s": both,
           "per_rank_send_GBps": round(12 * (world - 1) * block * 2 / comm_alone / 1e9, 2)}
    rows = [None] * world
    dist.all_gather_object(rows, row)
    if rank == 0:
        print(json.dumps(rows), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
