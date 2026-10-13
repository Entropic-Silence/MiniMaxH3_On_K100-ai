#!/usr/bin/env python3
"""Diagnostic: are the vendor SDPA kernels bitwise repeatable on Qwen3-VL shapes?"""
import torch
import torch.nn.functional as F


def check(name, make, repeats=30):
    torch.manual_seed(0)
    q, k, v, kwargs = make()
    first = F.scaled_dot_product_attention(q, k, v, **kwargs)
    diffs = sum(not torch.equal(first, F.scaled_dot_product_attention(q, k, v, **kwargs))
                for _ in range(repeats))
    print(f"{name}: {diffs}/{repeats} repeats differ", flush=True)


def main():
    bf = dict(device="cuda", dtype=torch.bfloat16)
    check("text causal GQA 64/8 heads, 1200 tokens",
          lambda: (torch.randn(1, 64, 1200, 128, **bf), torch.randn(1, 8, 1200, 128, **bf),
                   torch.randn(1, 8, 1200, 128, **bf), {"is_causal": True, "enable_gqa": True}))
    check("vision 16 heads d72, 2304 tokens",
          lambda: (torch.randn(1, 16, 2304, 72, **bf), torch.randn(1, 16, 2304, 72, **bf),
                   torch.randn(1, 16, 2304, 72, **bf), {}))
    check("DiT 14 heads, 22904 queries / 22901 keys",
          lambda: (torch.randn(1, 14, 22904, 128, **bf), torch.randn(1, 14, 22901, 128, **bf),
                   torch.randn(1, 14, 22901, 128, **bf), {}), repeats=10)


if __name__ == "__main__":
    main()
