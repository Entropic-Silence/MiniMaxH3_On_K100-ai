#!/usr/bin/env python3
"""Diagnostic: host-to-device weight upload rates on this machine.

Reads one checkpoint shard with the fast loader, then times the copy of a
quarter of it to each device concurrently (like a TP4 load) from pageable
memory, the cost of page-locking the buffer, and the same copy when pinned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../InfiniLM/python"))
from infinilm.modeling_utils import _read_safetensors  # noqa: E402


def upload(buffer, devices, repeats=2):
    part = buffer.numel() // len(devices)
    targets = [torch.empty(part, dtype=torch.uint8, device=f"cuda:{d}") for d in devices]

    def copy(i):
        torch.cuda.set_device(devices[i])
        targets[i].copy_(buffer[i * part:(i + 1) * part])
        torch.cuda.synchronize(devices[i])

    best = float("inf")
    with ThreadPoolExecutor(len(devices)) as pool:
        for _ in range(repeats):
            start = time.perf_counter()
            list(pool.map(copy, range(len(devices))))
            best = min(best, time.perf_counter() - start)
    return part * len(devices) / best / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    devices = [int(d) for d in args.gpus.split(",")]
    for d in devices:
        torch.empty(1, device=f"cuda:{d}")

    start = time.perf_counter()
    _, tensors = _read_safetensors(args.shard)
    read_s = time.perf_counter() - start
    buffer = next(iter(tensors.values())).untyped_storage()
    buffer = torch.tensor([], dtype=torch.uint8).set_(buffer)
    gb = buffer.numel() / 1e9
    print(f"read {gb:.2f} GB in {read_s:.2f} s ({gb / read_s:.2f} GB/s)", flush=True)

    print(f"pageable H2D x{len(devices)}: {upload(buffer, devices):.2f} GB/s", flush=True)
    single = upload(buffer[: buffer.numel() // len(devices)], devices[:1])
    print(f"pageable H2D x1: {single:.2f} GB/s", flush=True)

    start = time.perf_counter()
    status = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel(), 0)
    reg_s = time.perf_counter() - start
    print(f"hostRegister status {int(status)} {gb:.2f} GB in {reg_s:.2f} s ({gb / reg_s:.2f} GB/s)", flush=True)
    print(f"pinned H2D x{len(devices)}: {upload(buffer, devices):.2f} GB/s", flush=True)
    print(f"pinned H2D x1: {upload(buffer[: buffer.numel() // len(devices)], devices[:1]):.2f} GB/s", flush=True)
    torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())


if __name__ == "__main__":
    main()
