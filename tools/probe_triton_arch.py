#!/usr/bin/env python3
"""Platform smoke test for a new Hygon architecture: can Triton (and so the
NineToothed kernels of this tree) compile and run here, and which dtypes does
the device actually compute with?

It reports the device/arch, then runs, for every dtype that loads at all:
an elementwise kernel, a reduction, and a `tl.dot` (the matrix path, which on
gfx906/Z100 has no MFMA instruction and falls back to FMA). Each check prints
the relative error against the PyTorch result, so an architecture that
silently computes a dtype badly shows up here rather than inside a fused
kernel. Diagnostics only; never used by generation.
"""
import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(x + offsets, mask=mask) * tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def reduce_kernel(x, out, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    value = tl.load(x + row * width + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(out + row, tl.sum(value, 0))


@triton.jit
def dot_kernel(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        offs_k = start + tl.arange(0, BK)
        left = tl.load(a + offs_m[:, None] * K + offs_k[None, :], mask=offs_m[:, None] < M)
        right = tl.load(b + offs_k[:, None] * N + offs_n[None, :], mask=offs_n[None, :] < N)
        accumulator += tl.dot(left, right)
    tl.store(c + offs_m[:, None] * N + offs_n[None, :], accumulator,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def relative_error(out, reference):
    reference = reference.float()
    return ((out.float() - reference).norm() / reference.norm().clamp_min(1e-30)).item()


def check(name, fn):
    try:
        print(f"  {name:<22} rel err {fn():.3e}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<22} FAILED {type(exc).__name__}: {str(exc)[:140]}", flush=True)
        return False


def main() -> int:
    if not torch.cuda.is_available():
        print("no accelerator visible")
        return 1
    device = os.environ.get("PROBE_DEVICE", "cuda:0")
    properties = torch.cuda.get_device_properties(torch.device(device))
    print(f"device {properties.name}  arch {properties.gcnArchName}  "
          f"{properties.total_memory / 2**30:.1f} GiB  count {torch.cuda.device_count()}")
    print(f"torch {torch.__version__}  hip {torch.version.hip}  triton {triton.__version__}")
    print(f"triton backends {sorted(os.listdir(triton.__path__[0] + '/backends'))}")

    ok = True
    for name, dtype in (("float32", torch.float32), ("float16", torch.float16),
                        ("bfloat16", torch.bfloat16)):
        print(f"{name}:", flush=True)
        try:
            x = torch.randn(4096, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            print(f"  unsupported on this device: {type(exc).__name__}: {exc}")
            ok = False
            continue
        y = torch.randn_like(x)

        def elementwise():
            out = torch.empty_like(x)
            elementwise_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK=256)
            torch.cuda.synchronize()
            return relative_error(out, x * y)

        rows, width = 64, 512
        table = torch.randn(rows, width, device=device, dtype=dtype)

        def reduction():
            out = torch.empty(rows, device=device, dtype=torch.float32)
            reduce_kernel[(rows,)](table, out, width, BLOCK=width)
            torch.cuda.synchronize()
            return relative_error(out, table.float().sum(-1))

        size = 512
        left = torch.randn(size, size, device=device, dtype=dtype)
        right = torch.randn(size, size, device=device, dtype=dtype)

        def dot():
            out = torch.empty(size, size, device=device, dtype=torch.float32)
            dot_kernel[(size // 64, size // 64)](left, right, out, size, size, size,
                                                 BM=64, BN=64, BK=32)
            torch.cuda.synchronize()
            return relative_error(out, left.float() @ right.float())

        ok &= check("elementwise", elementwise)
        ok &= check("reduction", reduction)
        ok &= check("tl.dot", dot)
    print("ALL_OK" if ok else "SOME_CHECKS_FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
