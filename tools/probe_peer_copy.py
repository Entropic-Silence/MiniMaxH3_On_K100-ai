#!/usr/bin/env python3
"""Diagnostic: cross-device copy and event semantics on DTK HIP.

Checks, for hipMemcpyAsync(DeviceToDevice) and hipMemcpyPeerAsync between two
devices:
  * whether a stream on device A honours hipStreamWaitEvent on an event
    recorded on device B (copy must land after B's slow writer kernel);
  * whether the copy is done by a copy engine (no slowdown of a GEMM loop on
    the source device) or by a compute kernel.
"""
import ctypes
import time

import torch

hip = ctypes.CDLL("libamdhip64.so")
D2D = 3


def check(status, what):
    if status != 0:
        raise RuntimeError(f"{what}: HIP error {status}")


def event(device):
    check(hip.hipSetDevice(device), "hipSetDevice")
    e = ctypes.c_void_p()
    check(hip.hipEventCreateWithFlags(ctypes.byref(e), 2), "hipEventCreateWithFlags")  # disable timing
    return e


def copy(kind, dst, dst_dev, src, src_dev, nbytes, stream):
    if kind == "d2d":
        check(hip.hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes),
                                 D2D, ctypes.c_void_p(stream)), "hipMemcpyAsync")
    else:
        check(hip.hipMemcpyPeerAsync(ctypes.c_void_p(dst), dst_dev, ctypes.c_void_p(src), src_dev,
                                     ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)), "hipMemcpyPeerAsync")


def main():
    a, b = 1, 0
    for i in (a, b):
        check(hip.hipSetDevice(i), "hipSetDevice")
        status = hip.hipDeviceEnablePeerAccess(b if i == a else a, 0)
        assert status in (0, 704), status
    n = 64 << 20
    src = torch.full((n // 2,), 1.0, dtype=torch.bfloat16, device=f"cuda:{a}")
    dst = torch.zeros((n // 2,), dtype=torch.bfloat16, device=f"cuda:{b}")
    sa = torch.cuda.Stream(device=a)
    sb = torch.cuda.Stream(device=b)
    big = torch.randn(8192, 8192, device=f"cuda:{b}", dtype=torch.bfloat16)
    for kind in ("d2d", "peer"):
        ok = 0
        for trial in range(5):
            dst.zero_()
            torch.cuda.synchronize(b)
            ready = event(b)
            with torch.cuda.stream(sb):
                for _ in range(20):            # slow writer on B: ~40+ ms
                    torch.mm(big, big)
                dst.fill_(2.0)                 # B writes dst last
            check(hip.hipEventRecord(ready, ctypes.c_void_p(sb.cuda_stream)), "hipEventRecord")
            check(hip.hipStreamWaitEvent(ctypes.c_void_p(sa.cuda_stream), ready, 0), "hipStreamWaitEvent")
            check(hip.hipSetDevice(a), "hipSetDevice")
            copy(kind, dst.data_ptr(), b, src.data_ptr(), a, n, sa.cuda_stream)
            torch.cuda.synchronize(a)
            torch.cuda.synchronize(b)
            ok += int(bool((dst == 1.0).all()))
        print(f"{kind}: copy ordered after cross-device event in {ok}/5 trials", flush=True)

    # Copy engine or compute kernel? GEMM loop on the source device with and without copies.
    m = torch.randn(8192, 8192, device=f"cuda:{a}", dtype=torch.bfloat16)
    gs = torch.cuda.Stream(device=a)

    def gemms(k):
        with torch.cuda.stream(gs):
            for _ in range(k):
                torch.mm(m, m)

    gemms(2)
    torch.cuda.synchronize(a)
    t = time.perf_counter(); gemms(30); torch.cuda.synchronize(a); alone = time.perf_counter() - t
    for kind in ("d2d", "peer"):
        torch.cuda.synchronize(a)
        t = time.perf_counter()
        for _ in range(20):
            copy(kind, dst.data_ptr(), b, src.data_ptr(), a, n, sa.cuda_stream)
        torch.cuda.synchronize(a); torch.cuda.synchronize(b)
        copies = time.perf_counter() - t
        t = time.perf_counter()
        gemms(30)
        for _ in range(20):
            copy(kind, dst.data_ptr(), b, src.data_ptr(), a, n, sa.cuda_stream)
        torch.cuda.synchronize(a); torch.cuda.synchronize(b)
        both = time.perf_counter() - t
        print(f"{kind}: gemms {alone:.3f}s, copies {copies:.3f}s ({20 * n / copies / 1e9:.1f} GB/s), both {both:.3f}s",
              flush=True)


if __name__ == "__main__":
    main()
