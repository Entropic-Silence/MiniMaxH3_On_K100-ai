#!/usr/bin/env python3
"""Diagnostic: NineToothed h3_rope_cache vs the PyTorch minimax_h3_rotary_cache
(bitwise), on the captured astronaut positions and on synthetic ones."""
import sys

import torch

from h3_pointwise import make_rope_cache


def reference(position_ids, rope_freq_dim=16, rope_theta=10000.0):
    # Copy of infinilm.multimodal.minimax_h3.minimax_h3_rotary_cache (torch).
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32,
                                                  device=position_ids.device) / (2 * rope_freq_dim)))
    frequencies = (position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)).flatten(1)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(torch.bfloat16)


def power(rope_freq_dim=16, rope_theta=10000.0):
    # Stand-in for InfiniCore float_power (device powf), as in the reference.
    exponent = torch.tensor([2 * k / (2 * rope_freq_dim) for k in range(rope_freq_dim)],
                            dtype=torch.float32, device="cuda")
    return torch.full_like(exponent, rope_theta) ** exponent


def find_positions(obj):
    if isinstance(obj, dict):
        if "position_ids" in obj and torch.is_tensor(obj["position_ids"]):
            return obj["position_ids"]
        for value in obj.values():
            found = find_positions(value)
            if found is not None:
                return found
    return None


def check(name, kernel, positions):
    ref = reference(positions)
    out = torch.empty(positions.shape[0], 96, device="cuda", dtype=torch.bfloat16)
    kernel(positions.double().contiguous(), power(), out[:, :48], out[:, 48:])
    torch.cuda.synchronize()
    mismatch = (out != ref).sum().item()
    print(f"{name}: dtype {positions.dtype} shape {tuple(positions.shape)} "
          f"range [{positions.min().item():.4g}, {positions.max().item():.4g}] "
          f"bitwise {mismatch == 0} mismatches {mismatch}", flush=True)
    return mismatch == 0


def main():
    kernel = make_rope_cache()
    ok = True
    if len(sys.argv) > 1:
        positions = find_positions(torch.load(sys.argv[1], weights_only=True, map_location="cpu"))
        ok &= check("captured", kernel, positions.cuda())
    torch.manual_seed(0)
    ok &= check("integer", kernel, torch.randint(0, 4096, (50000, 3), device="cuda").float())
    ok &= check("uniform", kernel, torch.rand(50000, 3, device="cuda") * 2000 - 100)
    ok &= check("fine", kernel, torch.arange(150000, device="cuda").float().view(-1, 3) / 37)
    ok &= check("fine64", kernel, torch.arange(150000, device="cuda").double().view(-1, 3) / 37)
    ok &= check("int64", kernel, torch.randint(-5000, 5000, (50000, 3), device="cuda"))
    print("ALL_BITWISE" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
