#!/usr/bin/env python3
"""Diagnostic: BF16 GEMM throughput of the H3 SP4 per-card shapes under
PyTorch's BLAS backends (rocBLAS vs hipBLASLt)."""
import torch
import torch.nn.functional as F

SHAPES = {  # name: (M, N, K), output = x[M,K] @ w[N,K]^T
    "qkv": (5725, 16128, 5376),
    "qkv_group": (5725, 8064, 5376),
    "out": (5725, 5376, 5376),
    "gate_up": (5725, 28672, 5376),
    "down": (5725, 5376, 14336),
}


def timed(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    for backend in ("cublas", "cublaslt"):
        try:
            torch.backends.cuda.preferred_blas_library(backend)
        except Exception as exc:  # noqa: BLE001
            print(backend, "unavailable:", exc)
            continue
        for name, (m, n, k) in SHAPES.items():
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            ms = timed(lambda: F.linear(x, w))
            print(f"{backend:9s} {name:10s} {ms:7.2f} ms {2 * m * n * k / ms / 1e9:6.1f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()
