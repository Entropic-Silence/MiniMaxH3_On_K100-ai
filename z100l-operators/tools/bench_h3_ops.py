#!/usr/bin/env python3
"""Operator-level correctness and performance harness for the MiniMax-H3
NineToothed kernels of this tree (attention, q/k norm + RoPE, RoPE, the RoPE
table, RMSNorm, RMSNorm/AdaLN, gated residual, gated residual + RMSNorm/AdaLN,
SwiGLU).

For every operator it builds the kernel with NineToothed, runs the eager
PyTorch composition that the reference pipeline performs as the baseline, and
reports whether the kernel output is bitwise identical to it (plus the largest
absolute difference when it is not) and how long each takes. The reference
composition keeps the per-step BF16 rounding of the Diffusers implementation,
which is what these kernels reproduce.

It needs only PyTorch and NineToothed (no InfiniCore build), so the same
command runs on any Hygon architecture:

    python tools/bench_h3_ops.py --seq 8192 --heads 14 --json ops_z100.json

Options worth knowing:
    --dtype bf16|fp16   kernel and tensor dtype (gfx906/Z100 has no native
                        BF16 arithmetic, so fp16 is the fallback to compare)
    --ops a,b,c         run only these operators
    --repeats N         timed iterations (after 3 warmup iterations)
Never used by generation; diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "ninetoothed/src"),
                str(Path(__file__).resolve().parent)]

import ninetoothed  # noqa: E402

from h3_attention import make_attention  # noqa: E402
from h3_pointwise import (  # noqa: E402
    make_gated_residual,
    make_gated_residual_rms_modulate,
    make_qk_norm_rope,
    make_rms,
    make_rms_modulate,
    make_rope,
    make_rope_cache,
    make_swiglu,
)

DTYPES = {"bf16": (torch.bfloat16, ninetoothed.bfloat16),
          "fp16": (torch.float16, ninetoothed.float16)}
HIDDEN = 5376
FFN = 14336
HEAD_DIM = 128


def timed(fn, repeats: int) -> float:
    """Milliseconds per call, after three warmup calls."""
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


def difference(out: torch.Tensor, reference: torch.Tensor) -> tuple[bool, float]:
    same = torch.equal(out.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
    if same:
        return True, 0.0
    return False, (out.float() - reference.float()).abs().max().item()


def rms_norm_reference(x, weight, eps):
    """RMSNorm with each intermediate rounded to the tensor dtype, as the
    Diffusers reference does in BF16."""
    squared = (x.float() * x.float()).to(x.dtype)
    mean = (squared.float().sum(-1, keepdim=True) / x.shape[-1]).to(x.dtype)
    variance = (mean.float() + torch.tensor(eps, dtype=torch.float32, device=x.device)).to(x.dtype)
    reciprocal = torch.rsqrt(variance.float()).to(x.dtype)
    return ((x.float() * reciprocal.float()).to(x.dtype).float() * weight.float()).to(x.dtype)


def modulate_reference(normalized, scale, shift):
    factor = (scale.float() + 1.0).to(scale.dtype)
    return ((normalized.float() * factor.float()).to(normalized.dtype).float() + shift.float()).to(normalized.dtype)


def rope_reference(x, cos, sin):
    """3-axis NeoX RoPE on the first 96 of 128 channels, BF16 rounding per step."""
    a, b, passthrough = x[..., :48], x[..., 48:96], x[..., 96:]
    c = cos.float().unsqueeze(-2)
    s = sin.float().unsqueeze(-2)
    ac = (a.float() * c).to(x.dtype)
    bs = (-b.float() * s).to(x.dtype)
    bc = (b.float() * c).to(x.dtype)
    a_s = (a.float() * s).to(x.dtype)
    return torch.cat([(ac.float() + bs.float()).to(x.dtype),
                      (bc.float() + a_s.float()).to(x.dtype), passthrough], dim=-1)


def rotary_cache_reference(position_ids, rope_freq_dim=16, rope_theta=10000.0, dtype=torch.bfloat16):
    inverse = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32,
                                                 device=position_ids.device) / (2 * rope_freq_dim)))
    frequencies = (position_ids.float().unsqueeze(-1) * inverse.view(1, 1, -1)).flatten(1)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(dtype)


def op_attention(cfg):
    heads, seq, dtype = cfg.heads, cfg.seq, cfg.torch_dtype
    scale = HEAD_DIM ** -0.5
    q, k, v = (torch.randn(1, heads, seq, HEAD_DIM, device=cfg.device, dtype=dtype) for _ in range(3))
    kernel = make_attention(dtype=cfg.nt_dtype, **cfg.attention_config)
    out = torch.empty_like(q)
    baseline = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)  # noqa: E731
    reference = baseline()
    kernel(q, k, v, scale, out)
    flops = 4 * seq * seq * HEAD_DIM * heads
    return out, reference, lambda: kernel(q, k, v, scale, out), baseline, flops


def op_qk_norm_rope(cfg):
    """Per-head RMSNorm + 3-axis RoPE written into a packed send buffer, on a
    strided view of a fused QKV projection (the sequence-parallel layout)."""
    tokens, dtype = cfg.seq, cfg.torch_dtype
    ranks, heads = 1, cfg.heads
    projected = torch.randn(tokens, 3, ranks * heads, HEAD_DIM, device=cfg.device, dtype=dtype)
    source = projected[:, 0]
    x = source.as_strided((tokens, ranks, heads, HEAD_DIM),
                          (source.stride(0), heads * source.stride(1), source.stride(1), 1))
    weight = (torch.rand(HEAD_DIM, device=cfg.device) + 0.5).to(dtype)
    cos = torch.rand(tokens, 48, device=cfg.device).to(dtype)
    sin = torch.rand(tokens, 48, device=cfg.device).to(dtype)
    send = torch.empty(ranks, tokens, heads, HEAD_DIM, device=cfg.device, dtype=dtype)
    out = send.permute(1, 0, 2, 3)
    kernel = make_qk_norm_rope(dtype=cfg.nt_dtype)

    def baseline():
        normalized = rms_norm_reference(x, weight, 1e-6)
        rotated = rope_reference(normalized, cos.unsqueeze(1).expand(-1, ranks, -1),
                                 sin.unsqueeze(1).expand(-1, ranks, -1))
        return rotated.permute(1, 0, 2, 3).contiguous()

    reference = baseline()
    kernel(x, weight, 1e-6, cos, sin, out)
    return send, reference, lambda: kernel(x, weight, 1e-6, cos, sin, out), baseline, None


def op_rope(cfg):
    tokens, dtype = cfg.seq, cfg.torch_dtype
    x = torch.randn(tokens, 1, HEAD_DIM, device=cfg.device, dtype=dtype)
    cos = torch.rand(tokens, 48, device=cfg.device).to(dtype)
    sin = torch.rand(tokens, 48, device=cfg.device).to(dtype)
    kernel = make_rope(dtype=cfg.nt_dtype)
    out = torch.empty_like(x)
    baseline = lambda: rope_reference(x, cos, sin)  # noqa: E731
    reference = baseline()
    kernel(x, cos, sin, out)
    return out, reference, lambda: kernel(x, cos, sin, out), baseline, None


def op_rope_cache(cfg):
    tokens = cfg.seq
    positions = torch.randint(0, 4096, (tokens, 3), device=cfg.device).double()
    exponent = torch.tensor([2 * k / 32 for k in range(16)], dtype=torch.float32, device=cfg.device)
    power = torch.full_like(exponent, 10000.0) ** exponent
    kernel = make_rope_cache(dtype=cfg.nt_dtype)
    out = torch.empty(tokens, 96, device=cfg.device, dtype=cfg.torch_dtype)
    baseline = lambda: rotary_cache_reference(positions, dtype=cfg.torch_dtype)  # noqa: E731
    reference = baseline()
    kernel(positions, power, out[:, :48], out[:, 48:])
    return out, reference, lambda: kernel(positions, power, out[:, :48], out[:, 48:]), baseline, None


def op_rms(cfg):
    tokens, dtype = cfg.seq, cfg.torch_dtype
    x = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype) * 8
    weight = torch.randn(HIDDEN, device=cfg.device, dtype=dtype)
    kernel = make_rms(width=HIDDEN, dtype=cfg.nt_dtype)
    out = torch.empty_like(x)
    baseline = lambda: rms_norm_reference(x, weight, 1e-6)  # noqa: E731
    reference = baseline()
    kernel(x, weight, 1e-6, out)
    return out, reference, lambda: kernel(x, weight, 1e-6, out), baseline, None


def op_rms_modulate(cfg):
    tokens, dtype = cfg.seq, cfg.torch_dtype
    x = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype) * 8
    weight = torch.randn(HIDDEN, device=cfg.device, dtype=dtype)
    modulation_tokens = 1 if getattr(cfg, "broadcast_modulation", False) else tokens
    scale = (torch.randn(1, modulation_tokens, HIDDEN, device=cfg.device, dtype=dtype) * 0.5).expand(1, tokens, HIDDEN)
    shift = (torch.randn(1, modulation_tokens, HIDDEN, device=cfg.device, dtype=dtype) * 0.5).expand(1, tokens, HIDDEN)
    kernel = make_rms_modulate(dtype=cfg.nt_dtype)
    out = torch.empty_like(x)
    baseline = lambda: modulate_reference(rms_norm_reference(x, weight, 1e-6), scale, shift)  # noqa: E731
    reference = baseline()
    kernel(x, weight, 1e-6, scale, shift, out)
    return out, reference, lambda: kernel(x, weight, 1e-6, scale, shift, out), baseline, None


def op_gated_residual(cfg):
    tokens, dtype = cfg.seq, cfg.torch_dtype
    x = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype) * 8
    y = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype)
    gate = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype) * 0.5
    kernel = make_gated_residual(dtype=cfg.nt_dtype)
    out = torch.empty_like(x)
    baseline = lambda: torch.add(x, torch.mul(gate, y))  # noqa: E731
    reference = baseline()
    kernel(x, gate, y, out)
    return out, reference, lambda: kernel(x, gate, y, out), baseline, None


def op_gated_residual_rms_modulate(cfg):
    tokens, dtype = cfg.seq, cfg.torch_dtype
    x = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype) * 8
    y = torch.randn(1, tokens, HIDDEN, device=cfg.device, dtype=dtype)
    modulation_tokens = 1 if getattr(cfg, "broadcast_modulation", False) else tokens
    gate = (torch.randn(1, modulation_tokens, HIDDEN, device=cfg.device, dtype=dtype) * 0.5).expand(1, tokens, HIDDEN)
    weight = torch.randn(HIDDEN, device=cfg.device, dtype=dtype)
    scale = (torch.randn(1, modulation_tokens, HIDDEN, device=cfg.device, dtype=dtype) * 0.5).expand(1, tokens, HIDDEN)
    shift = (torch.randn(1, modulation_tokens, HIDDEN, device=cfg.device, dtype=dtype) * 0.5).expand(1, tokens, HIDDEN)
    kernel = make_gated_residual_rms_modulate(dtype=cfg.nt_dtype)
    residual, out = torch.empty_like(x), torch.empty_like(x)

    def baseline():
        added = torch.add(x, torch.mul(gate, y))
        return added, modulate_reference(rms_norm_reference(added, weight, 1e-6), scale, shift)

    reference_residual, reference_out = baseline()
    kernel(x, gate, y, weight, 1e-6, scale, shift, residual, out)
    same_residual, delta_residual = difference(residual, reference_residual)
    if not same_residual:
        raise AssertionError(f"residual differs, max |diff| {delta_residual:.3g}")
    return out, reference_out, lambda: kernel(x, gate, y, weight, 1e-6, scale, shift, residual, out), baseline, None


def op_swiglu(cfg):
    tokens, dtype = cfg.seq, cfg.torch_dtype
    packed = torch.randn(1, tokens, 2 * FFN, device=cfg.device, dtype=dtype) * 2
    up, gate = packed[..., :FFN], packed[..., FFN:]
    kernel = make_swiglu(dtype=cfg.nt_dtype)
    out = torch.empty(1, tokens, FFN, device=cfg.device, dtype=dtype)

    def baseline():
        value = gate.float()
        activated = (value / (1.0 + torch.exp(-value))).to(dtype)
        return up * activated

    reference = baseline()
    kernel(up, gate, out)
    return out, reference, lambda: kernel(up, gate, out), baseline, None


OPERATORS = {
    "attention": op_attention,
    "qk_norm_rope": op_qk_norm_rope,
    "rope": op_rope,
    "rope_cache": op_rope_cache,
    "rms": op_rms,
    "rms_modulate": op_rms_modulate,
    "gated_residual": op_gated_residual,
    "gated_residual_rms_modulate": op_gated_residual_rms_modulate,
    "swiglu": op_swiglu,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=int, default=8192, help="tokens (default 8192)")
    parser.add_argument("--heads", type=int, default=14, help="attention heads per device")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ops", help="comma-separated subset of " + ",".join(OPERATORS))
    parser.add_argument("--attention-config", default="{}",
                        help='JSON overrides, e.g. {"block_m": 128, "block_n": 32, "num_warps": 4}')
    parser.add_argument("--json", type=Path, help="write the results as JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no accelerator visible")
        return 1
    torch.manual_seed(args.seed)
    torch_dtype, nt_dtype = DTYPES[args.dtype]
    attention_config = {"block_m": 128, "block_n": 32, "num_warps": 4, "num_stages": 1}
    attention_config.update(json.loads(args.attention_config))

    class Config:
        seq = args.seq
        heads = args.heads
        device = args.device
        torch_dtype = None
        nt_dtype = None
        attention_config = None

    cfg = Config()
    cfg.torch_dtype = torch_dtype
    cfg.nt_dtype = nt_dtype
    cfg.attention_config = attention_config

    properties = torch.cuda.get_device_properties(torch.device(args.device))
    environment = {"device": properties.name, "arch": properties.gcnArchName,
                   "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                   "dtype": args.dtype, "seq": args.seq, "heads": args.heads,
                   "attention_config": attention_config}
    print(json.dumps(environment), flush=True)
    print(f"{'operator':<30} {'kernel':>10} {'eager':>10} {'speedup':>8} {'bitwise':>8}  note", flush=True)

    selected = args.ops.split(",") if args.ops else list(OPERATORS)
    results = []
    for name in selected:
        if name not in OPERATORS:
            print(f"{name:<30} unknown operator")
            continue
        try:
            with torch.cuda.device(args.device), torch.inference_mode():
                out, reference, run, baseline, flops = OPERATORS[name](cfg)
                same, delta = difference(out, reference)
                kernel_ms = timed(run, args.repeats)
                eager_ms = timed(baseline, args.repeats)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<30} failed: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            results.append({"operator": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        note = "" if same else f"max |diff| {delta:.3g}"
        if flops:
            note = (note + " " if note else "") + f"{flops / kernel_ms / 1e9:.1f} TFLOPS"
        print(f"{name:<30} {kernel_ms:9.3f}m {eager_ms:9.3f}m {eager_ms / kernel_ms:7.2f}x "
              f"{str(same):>8}  {note}", flush=True)
        results.append({"operator": name, "kernel_ms": kernel_ms, "eager_ms": eager_ms,
                        "speedup": eager_ms / kernel_ms, "bitwise_equal": same,
                        "max_abs_difference": delta, "flops": flops})
    if args.json:
        args.json.write_text(json.dumps({"environment": environment, "results": results,
                                         "diagnostic_only": True}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
