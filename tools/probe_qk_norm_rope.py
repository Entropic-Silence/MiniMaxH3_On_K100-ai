#!/usr/bin/env python3
"""Diagnostic: fused q/k RMSNorm + RoPE + pack vs h3_rms_exact then h3_rope_exact."""
import torch

from h3_pointwise import make_qk_norm_rope, make_rms, make_rope


def timed(fn, repeats=20):
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
    torch.manual_seed(0)
    tokens, ranks, heads, d = 5725, 4, 7, 128
    n = ranks * heads
    projected = torch.randn(tokens, 3, n, d, device="cuda", dtype=torch.bfloat16) * 3
    x = projected[:, 0]                                  # [T, n, D], row stride 3nD
    x4 = x.view(tokens, ranks, heads, d)                 # [T, P, Hs, D]
    weight = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    table = torch.randn(tokens, 96, device="cuda", dtype=torch.bfloat16)
    cos, sin = table[:, :48], table[:, 48:]
    send = torch.empty(ranks, tokens, heads, d, device="cuda", dtype=torch.bfloat16)
    out4 = send.permute(1, 0, 2, 3)                      # [T, P, Hs, D] view of [P, T, Hs, D]

    rms, rope = make_rms(width=128), make_rope()
    normed = torch.empty(tokens, n, d, device="cuda", dtype=torch.bfloat16)
    rms(x, weight, 1e-5, normed)
    rotated = torch.empty_like(normed)
    rope(normed, cos, sin, rotated)
    reference = rotated.view(tokens, ranks, heads, d).permute(1, 0, 2, 3).contiguous()

    for rows, warps in ((8, 4), (8, 2), (4, 2), (16, 4)):
        fused = make_qk_norm_rope(rows=rows, num_warps=warps)
        send.zero_()
        fused(x4, weight, 1e-5, cos, sin, out4)
        same = torch.equal(send, reference)
        diff = (send.float() - reference.float()).abs().max().item()
        ms = timed(lambda: fused(x4, weight, 1e-5, cos, sin, out4))
        print(f"rows={rows} warps={warps}: bitwise {same} max|diff| {diff:.3g} {ms:.3f} ms", flush=True)

    def unfused():
        rms(x, weight, 1e-5, normed)
        rope(normed, cos, sin, rotated)
        rotated.view(tokens, ranks, heads, d).permute(1, 0, 2, 3).contiguous()
    print(f"unfused rms+rope+pack: {timed(unfused):.3f} ms", flush=True)


if __name__ == "__main__":
    main()
