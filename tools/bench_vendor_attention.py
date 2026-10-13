#!/usr/bin/env python3
"""Diagnostic: vendor attention kernels on the H3 per-card shape (gfx928)."""
import torch
import torch.nn.functional as F


def timed(fn, repeats=10):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats / 1e3


def main():
    s, h, d = 22901, 14, 128
    q, k, v = (torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flops = 4 * s * s * h * d
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    sdpa = timed(lambda: F.scaled_dot_product_attention(qt, kt, vt))
    print(f"torch SDPA (BHSD views): {sdpa * 1e3:.1f} ms, {flops / sdpa / 1e12:.1f} TFLOPS", flush=True)
    for backend in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(getattr(SDPBackend, backend)):
                t = timed(lambda: F.scaled_dot_product_attention(qt, kt, vt), repeats=3)
            print(f"  SDPA {backend}: {t * 1e3:.1f} ms, {flops / t / 1e12:.1f} TFLOPS", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  SDPA {backend}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
    try:
        from flash_attn import flash_attn_func
        t = timed(lambda: flash_attn_func(q, k, v))
        print(f"flash_attn_func (BSHD): {t * 1e3:.1f} ms, {flops / t / 1e12:.1f} TFLOPS", flush=True)
        ref = F.scaled_dot_product_attention(qt, kt, vt).transpose(1, 2)
        out = flash_attn_func(q, k, v)
        print("flash vs SDPA max abs", float((out.float() - ref.float()).abs().max()), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"flash_attn_func: {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    main()
