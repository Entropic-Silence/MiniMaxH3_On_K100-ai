#!/usr/bin/env python3
"""Baseline measurement of the MiniMax-H3 operators as InfiniCore implements
them, against the PyTorch composition the reference pipeline performs.

Unlike tools/bench_h3_ops.py (which measures this tree's NineToothed kernels
and therefore needs a Triton backend for the architecture), this harness only
needs a built InfiniCore with its Python extension, so it runs on any
architecture InfiniCore supports -- including gfx906/Z100, where upstream
Triton has no backend.

For every operator it reports whether InfiniCore's result matches the PyTorch
reference (bitwise, and the largest absolute difference otherwise) and the
time of each, at the H3 shapes. Use it to establish the baseline an
optimization is measured against, and to re-measure afterwards.

    python tools/bench_infinicore_ops.py --tokens 8192 --json base.json

Diagnostics only; never used by generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import infinicore
from infinicore.lib import _infinicore

HIDDEN = 5376
FFN = 14336
HEAD_DIM = 128
TIME_EMBED_DIM = 256


def wrap(tensor: torch.Tensor):
    """An InfiniCore view of a torch tensor's storage (no copy)."""
    return infinicore.strided_from_blob(
        tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
        dtype=infinicore.utils.to_infinicore_dtype(tensor.dtype),
        device=infinicore.device(tensor.device.type, tensor.device.index or 0),
    )._underlying


def timed(fn, repeats: int, device) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / repeats


def difference(out: torch.Tensor, reference: torch.Tensor):
    if torch.equal(out, reference):
        return True, 0.0
    return False, (out.float() - reference.float()).abs().max().item()


def op_rms_norm(cfg):
    x = torch.randn(1, cfg.tokens, HIDDEN, device=cfg.device, dtype=cfg.dtype) * 8
    weight = torch.randn(HIDDEN, device=cfg.device, dtype=cfg.dtype)
    out = torch.empty_like(x)
    native = lambda: _infinicore.rms_norm_(wrap(out), wrap(x), wrap(weight), 1e-6)  # noqa: E731
    reference = lambda: torch.nn.functional.rms_norm(x, (HIDDEN,), weight, eps=1e-6)  # noqa: E731
    native()
    return out, reference(), native, reference, None


def op_swiglu(cfg):
    packed = torch.randn(1, cfg.tokens, 2 * FFN, device=cfg.device, dtype=cfg.dtype) * 2
    up, gate = packed[..., :FFN].contiguous(), packed[..., FFN:].contiguous()
    out = torch.empty(1, cfg.tokens, FFN, device=cfg.device, dtype=cfg.dtype)
    native = lambda: _infinicore.swiglu_(wrap(out), wrap(up), wrap(gate))  # noqa: E731
    reference = lambda: up * torch.nn.functional.silu(gate)  # noqa: E731
    native()
    return out, reference(), native, reference, None


def op_gemm(cfg):
    """The DiT's widest projection shape (gate_up), as InfiniCore's linear."""
    m, n, k = cfg.tokens, 2 * FFN // cfg.shard, HIDDEN
    a = torch.randn(m, k, device=cfg.device, dtype=cfg.dtype) * 0.05
    b = torch.randn(n, k, device=cfg.device, dtype=cfg.dtype) * 0.05
    out = torch.empty(m, n, device=cfg.device, dtype=cfg.dtype)
    native = lambda: _infinicore.matmul_(wrap(out), wrap(a), wrap(b.t()))  # noqa: E731
    reference = lambda: a @ b.t()  # noqa: E731
    native()
    return out, reference(), native, reference, 2 * m * n * k


def op_attention(cfg):
    heads = cfg.heads
    q, k, v = (torch.randn(1, cfg.tokens, heads, HEAD_DIM, device=cfg.device, dtype=cfg.dtype)
               for _ in range(3))
    out = torch.empty_like(q)
    scale = HEAD_DIM ** -0.5

    def native():
        _infinicore.mha_(wrap(out), wrap(q), wrap(k), wrap(v), None, scale, False)

    def reference():
        return torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
            scale=scale).permute(0, 2, 1, 3).contiguous()

    native()
    return out, reference(), native, reference, 4 * cfg.tokens * cfg.tokens * HEAD_DIM * heads


def op_timestep_embedding(cfg):
    timestep = torch.rand(2, device=cfg.device, dtype=torch.float32) * 1000
    out = torch.empty(2, TIME_EMBED_DIM, device=cfg.device, dtype=torch.float32)
    native = lambda: _infinicore.timestep_embedding_(wrap(out), wrap(timestep))  # noqa: E731

    def reference():
        half = TIME_EMBED_DIM // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0, device=cfg.device))
            * torch.arange(half, device=cfg.device, dtype=torch.float32) / half)
        angles = timestep[:, None] * frequencies[None, :]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    native()
    return out, reference(), native, reference, None


def op_embedding(cfg):
    """Row gather, as the AdaLN modulation and the RoPE table use it."""
    table = torch.randn(3 * 2, 6 * HIDDEN, device=cfg.device, dtype=cfg.dtype)
    indices = torch.randint(0, table.shape[0], (cfg.tokens,), device=cfg.device)
    out = torch.empty(cfg.tokens, table.shape[1], device=cfg.device, dtype=cfg.dtype)
    native = lambda: _infinicore.embedding_(wrap(out), wrap(indices), wrap(table))  # noqa: E731
    reference = lambda: table[indices]  # noqa: E731
    native()
    return out, reference(), native, reference, None


OPERATORS = {
    "rms_norm": op_rms_norm,
    "swiglu": op_swiglu,
    "gemm_gate_up": op_gemm,
    "attention": op_attention,
    "timestep_embedding": op_timestep_embedding,
    "embedding": op_embedding,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--shard", type=int, default=1,
                        help="divide the GEMM's N by this (a tensor-parallel shard)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ops", help="comma-separated subset of " + ",".join(OPERATORS))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no accelerator visible")
        return 1
    torch.manual_seed(args.seed)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    class Config:
        tokens = args.tokens
        heads = args.heads
        shard = args.shard
        device = args.device
        dtype = dtypes[args.dtype]

    cfg = Config()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    infinicore.set_device(infinicore.device(device.type, device.index or 0))

    properties = torch.cuda.get_device_properties(device)
    environment = {"device": properties.name, "arch": properties.gcnArchName,
                   "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                   "dtype": args.dtype, "tokens": args.tokens, "heads": args.heads,
                   "shard": args.shard}
    print(json.dumps(environment), flush=True)
    print(f"{'operator':<22} {'infinicore':>11} {'torch':>10} {'ratio':>7} {'match':>7}  note",
          flush=True)

    results = []
    for name in (args.ops.split(",") if args.ops else list(OPERATORS)):
        if name not in OPERATORS:
            print(f"{name:<22} unknown operator")
            continue
        try:
            with torch.inference_mode():
                out, reference, native, torch_fn, flops = OPERATORS[name](cfg)
                same, delta = difference(out, reference)
                native_ms = timed(native, args.repeats, device)
                torch_ms = timed(torch_fn, args.repeats, device)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<22} failed: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            results.append({"operator": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        note = "" if same else f"max |diff| {delta:.3g}"
        if flops:
            note = (note + " " if note else "") + f"{flops / native_ms / 1e9:.1f} TFLOPS"
        print(f"{name:<22} {native_ms:10.3f}m {torch_ms:9.3f}m {torch_ms / native_ms:6.2f}x "
              f"{str(same):>7}  {note}", flush=True)
        results.append({"operator": name, "infinicore_ms": native_ms, "torch_ms": torch_ms,
                        "ratio": torch_ms / native_ms, "bitwise_equal": same,
                        "max_abs_difference": delta, "flops": flops})
    if args.json:
        args.json.write_text(json.dumps({"environment": environment, "results": results,
                                         "diagnostic_only": True}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
