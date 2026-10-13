#!/usr/bin/env python3
"""Peer-to-peer copy-engine (SDMA) bandwidth and its interference with compute.

Single process, all visible devices. Measures
  1. hipMemcpyPeerAsync bandwidth per ordered device pair (one copy at a time);
  2. an all-to-all pattern (every device sends one block to every peer at once);
  3. GEMM time on all devices alone, while the copy engines run the all-to-all
     pattern, so it can be compared with RCCL all-to-all (bench_rccl_interference.py).
Diagnostic only; not used by the generation path.
"""
import argparse
import ctypes
import json
import time

import torch

hip = ctypes.CDLL("libamdhip64.so")


def check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with HIP error {status}")


def enable_peer_access(devices):
    access = {}
    for i in devices:
        check(hip.hipSetDevice(i), "hipSetDevice")
        for j in devices:
            if i == j:
                continue
            can = ctypes.c_int(0)
            check(hip.hipDeviceCanAccessPeer(ctypes.byref(can), i, j), "hipDeviceCanAccessPeer")
            access[f"{i}->{j}"] = bool(can.value)
            if can.value:
                status = hip.hipDeviceEnablePeerAccess(j, 0)
                if status not in (0, 704):  # 704: already enabled
                    raise RuntimeError(f"hipDeviceEnablePeerAccess({i}->{j}) = {status}")
    return access


def peer_copy(dst, dst_dev, src, src_dev, nbytes, stream):
    check(hip.hipMemcpyPeerAsync(ctypes.c_void_p(dst.data_ptr()), dst_dev,
                                 ctypes.c_void_p(src.data_ptr()), src_dev,
                                 ctypes.c_size_t(nbytes), ctypes.c_void_p(stream.cuda_stream)),
          "hipMemcpyPeerAsync")


def sync_all(devices):
    for d in devices:
        torch.cuda.synchronize(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-mib", default="10,31,100")
    parser.add_argument("--gemm", default="5728,10752,5376")  # SP4 QKV group shape (M,N,K)
    parser.add_argument("--output")
    args = parser.parse_args()
    devices = list(range(torch.cuda.device_count()))
    result = {"devices": devices, "peer_access": enable_peer_access(devices)}
    streams = {d: torch.cuda.Stream(device=d) for d in devices}

    # 1. Pairwise bandwidth.
    nbytes = 256 << 20
    bufs = {d: (torch.empty(nbytes, dtype=torch.uint8, device=d),
                torch.empty(nbytes, dtype=torch.uint8, device=d)) for d in devices}
    pair = {}
    for i in devices:
        for j in devices:
            if i == j:
                continue
            peer_copy(bufs[j][1], j, bufs[i][0], i, nbytes, streams[i])
            sync_all(devices)
            start = time.perf_counter()
            for _ in range(5):
                peer_copy(bufs[j][1], j, bufs[i][0], i, nbytes, streams[i])
            sync_all(devices)
            pair[f"{i}->{j}"] = round(5 * nbytes / (time.perf_counter() - start) / 1e9, 2)
    result["pair_GBps"] = pair
    print(json.dumps({"pair_GBps": pair}), flush=True)

    # 2. All-to-all pattern: device i sends block b to every peer j concurrently.
    def all_to_all(src, dst, block):
        for i in devices:
            for j in devices:
                if i != j:
                    peer_copy(dst[j][i], j, src[i][j], i, block, streams[i])

    a2a = {}
    for mib in map(float, args.block_mib.split(",")):
        block = int(mib * (1 << 20))
        src = {i: {j: torch.empty(block, dtype=torch.uint8, device=i) for j in devices} for i in devices}
        dst = {j: {i: torch.empty(block, dtype=torch.uint8, device=j) for i in devices} for j in devices}
        all_to_all(src, dst, block)
        sync_all(devices)
        start = time.perf_counter()
        for _ in range(5):
            all_to_all(src, dst, block)
        sync_all(devices)
        seconds = (time.perf_counter() - start) / 5
        a2a[str(mib)] = {"seconds": seconds,
                         "per_rank_send_GBps": round((len(devices) - 1) * block / seconds / 1e9, 2)}
        print(json.dumps({"all_to_all_block_MiB": mib, **a2a[str(mib)]}), flush=True)
        del src, dst
    result["all_to_all"] = a2a

    # 3. GEMM interference.
    m, n, k = map(int, args.gemm.split(","))
    mats = {d: (torch.randn(m, k, device=d, dtype=torch.bfloat16),
                torch.randn(n, k, device=d, dtype=torch.bfloat16)) for d in devices}
    gemm_streams = {d: torch.cuda.Stream(device=d) for d in devices}

    def gemms(reps):
        for d in devices:
            with torch.cuda.device(d), torch.cuda.stream(gemm_streams[d]):
                a, b = mats[d]
                for _ in range(reps):
                    torch.mm(a, b.t())

    reps = 40
    gemms(2)
    sync_all(devices)
    start = time.perf_counter()
    gemms(reps)
    sync_all(devices)
    alone = (time.perf_counter() - start) / reps
    block = 31 << 20
    src = {i: {j: torch.empty(block, dtype=torch.uint8, device=i) for j in devices} for i in devices}
    dst = {j: {i: torch.empty(block, dtype=torch.uint8, device=j) for i in devices} for j in devices}
    sync_all(devices)
    start = time.perf_counter()
    gemms(reps)
    for _ in range(12):
        all_to_all(src, dst, block)
    sync_all(devices)
    both = time.perf_counter() - start
    # Copies alone for the same amount, to see how much overlapped.
    start = time.perf_counter()
    for _ in range(12):
        all_to_all(src, dst, block)
    sync_all(devices)
    copies = time.perf_counter() - start
    flops = 2 * m * n * k
    result["gemm_interference"] = {
        "gemm_ms_alone": alone * 1e3, "gemm_TFLOPS_alone": flops / alone / 1e12,
        "gemms_plus_copies_s": both, "gemms_alone_s": alone * reps, "copies_alone_s": copies}
    print(json.dumps(result["gemm_interference"]), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
