#!/usr/bin/env python3
"""Diagnostic: parallel pread throughput for checkpoint shards (no GPU)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np


def read_file(path, threads, chunk):
    size = os.path.getsize(path)
    buffer = np.empty(size, dtype=np.uint8)
    view = memoryview(buffer)
    fd = os.open(path, os.O_RDONLY)
    try:
        def read_range(offset):
            end = min(offset + chunk, size)
            while offset < end:
                count = os.preadv(fd, [view[offset:end]], offset)
                if count <= 0:
                    raise OSError("short read")
                offset += count
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(read_range, range(0, size, chunk)))
    finally:
        os.close(fd)
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--files-in-flight", type=int, default=1)
    parser.add_argument("--chunk-mb", type=int, default=64)
    args = parser.parse_args()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.files_in_flight) as pool:
        total = sum(pool.map(lambda p: read_file(p, args.threads, args.chunk_mb << 20), args.files))
    seconds = time.perf_counter() - start
    print(f"files={len(args.files)} in_flight={args.files_in_flight} threads={args.threads} "
          f"chunk={args.chunk_mb}MB: {total / 1e9:.1f} GB in {seconds:.1f} s = {total / 1e9 / seconds:.2f} GB/s", flush=True)


if __name__ == "__main__":
    main()
