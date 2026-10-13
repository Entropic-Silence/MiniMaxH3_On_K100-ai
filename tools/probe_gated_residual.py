#!/usr/bin/env python3
"""Diagnostic: the fused gated-residual kernels against the unfused ops on
H3-sized tensors (bitwise), and their time."""
import torch

from h3_pointwise import make_gated_residual, make_gated_residual_rms_modulate, make_rms_modulate


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
    rows, width = 22901, 5376
    x = torch.randn(1, rows, width, device="cuda", dtype=torch.bfloat16) * 8
    y = torch.randn(1, rows, width, device="cuda", dtype=torch.bfloat16)
    table = torch.randn(1, rows, width * 6, device="cuda", dtype=torch.bfloat16) * 0.5
    shift, scale, gate = table[..., :width], table[..., width:2 * width], table[..., 2 * width:3 * width]
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    residual_kernel = make_gated_residual()
    fused_kernel = make_gated_residual_rms_modulate()
    modulate_kernel = make_rms_modulate()

    reference = torch.add(x, torch.mul(gate, y))
    fused_residual = torch.empty_like(x)
    residual_kernel(x, gate, y, fused_residual)
    print("gated residual bitwise:", torch.equal(fused_residual, reference), flush=True)

    reference_out = torch.empty_like(x)
    modulate_kernel(reference, w, 1e-6, scale, shift, reference_out)
    residual, out = torch.empty_like(x), torch.empty_like(x)
    fused_kernel(x, gate, y, w, 1e-6, scale, shift, residual, out)
    print("fused residual bitwise:", torch.equal(residual, reference),
          "modulated bitwise:", torch.equal(out, reference_out), flush=True)

    unfused = timed(lambda: modulate_kernel(torch.add(x, torch.mul(gate, y)), w, 1e-6, scale, shift, reference_out))
    fused = timed(lambda: fused_kernel(x, gate, y, w, 1e-6, scale, shift, residual, out))
    plain = timed(lambda: torch.add(x, torch.mul(gate, y)))
    single = timed(lambda: residual_kernel(x, gate, y, fused_residual))
    print(f"residual+norm+modulate: unfused {unfused:.2f} ms, fused {fused:.2f} ms", flush=True)
    print(f"gated residual: torch mul+add {plain:.2f} ms, fused {single:.2f} ms", flush=True)
    for warps in (8, 16):
        kernel = make_gated_residual_rms_modulate(num_warps=warps)
        kernel(x, gate, y, w, 1e-6, scale, shift, residual, out)
        same = torch.equal(out, reference_out) and torch.equal(residual, reference)
        print(f"fused norm warps={warps}: {timed(lambda: kernel(x, gate, y, w, 1e-6, scale, shift, residual, out)):.2f} ms bitwise {same}", flush=True)
    for block in ((1, 1024), (1, 2048), (2, 1024), (4, 512), (8, 256), (4, 1024), (8, 512), (32, 128)):
        for warps in (4, 8):
            kernel = make_gated_residual(block=block, num_warps=warps)
            kernel(x, gate, y, fused_residual)
            same = torch.equal(fused_residual, reference)
            print(f"residual block={block} warps={warps}: {timed(lambda: kernel(x, gate, y, fused_residual)):.2f} ms bitwise {same}", flush=True)


if __name__ == "__main__":
    main()
