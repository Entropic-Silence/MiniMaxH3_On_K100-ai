#!/usr/bin/env python3
"""Diagnostic: page-lock a shared file mapping (page-cache pages) in place
instead of reading the file into anonymous memory first."""
import mmap
import os
import sys
import time

import torch


def upload(buffer, devices):
    part = buffer.numel() // len(devices)
    outs = [torch.empty(part, dtype=torch.uint8, device=f"cuda:{d}") for d in devices]
    start = time.perf_counter()
    for i, d in enumerate(devices):
        outs[i].copy_(buffer[i * part:(i + 1) * part], non_blocking=True)
    for d in devices:
        torch.cuda.synchronize(d)
    return buffer.numel() / (time.perf_counter() - start) / 1e9, outs


def main():
    path = sys.argv[1]
    devices = [0, 1, 2, 3]
    for d in devices:
        torch.empty(1, device=f"cuda:{d}")
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDWR)
    start = time.perf_counter()
    mapped = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    buffer = torch.frombuffer(mapped, dtype=torch.uint8)
    status = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), size, 0)
    reg = time.perf_counter() - start
    print(f"map+register {size / 1e9:.1f} GB: status {int(status)} in {reg:.2f} s ({size / 1e9 / reg:.1f} GB/s)", flush=True)
    rate, outs = upload(buffer, devices)
    print(f"H2D x4 from registered mapping: {rate:.1f} GB/s", flush=True)
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    same = bytes(outs[0][: 1 << 20].cpu().numpy()) == head
    print("first MiB identical:", same, flush=True)
    torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
    del buffer
    mapped.close()
    os.close(fd)


if __name__ == "__main__":
    main()
