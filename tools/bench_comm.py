#!/usr/bin/env python3
"""torchrun --standalone --nproc_per_node=2 tools/bench_comm.py --output comm.json"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", default="1,16,128,512")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    records = []
    try:
        for mib in map(int, args.sizes_mib.split(",")):
            n = mib * 1024 * 1024 // 4 // world * world
            for op in ["all_reduce", "all_gather", "all_to_all"]:
                src = torch.full((n,), rank + 1.0, device="cuda")
                dst = torch.empty(n * world if op == "all_gather" else n, device="cuda")
                def run():
                    if op == "all_reduce":
                        dist.all_reduce(src)
                    elif op == "all_gather":
                        dist.all_gather_into_tensor(dst, src)
                    else:
                        dist.all_to_all_single(dst, src)
                run()
                torch.cuda.synchronize()
                if op == "all_reduce":
                    assert (src == world * (world + 1) / 2).all().item()
                else:
                    chunks = dst.chunk(world)
                    assert all((chunk == i+1).all().item() for i, chunk in enumerate(chunks))
                samples = []
                for trial in range(2):
                    src.fill_(rank + 1.0)
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(5):
                        run()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) / 5)
                times = [None] * world
                dist.all_gather_object(times, samples)
                second = max(t[1] for t in times)
                # Algorithm payload bandwidth, not NCCL's bus bandwidth metric.
                row = {"operation": op, "ranks": world, "input_bytes_per_rank": n*4,
                       "seconds_by_rank": times, "second_run_max_seconds": second,
                       "payload_GBps": n*4 / second / 1e9, "correct": True}
                records.append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
                del src, dst
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"torch": torch.__version__, "hip": torch.version.hip,
                "arch": torch.cuda.get_device_properties(0).gcnArchName,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "records": records}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
