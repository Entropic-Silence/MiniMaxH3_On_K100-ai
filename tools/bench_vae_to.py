#!/usr/bin/env python3
"""Diagnostic: cost of the entry's `vae.to(device, bf16)` and whether moving
the (fp32) VAE to the device first and casting there gives identical tensors."""
import copy
import sys
import time

import torch
from diffusers import ModularPipeline


def main():
    modular_dir = sys.argv[1]
    print("torch threads", torch.get_num_threads(), flush=True)
    pipe = ModularPipeline.from_pretrained(modular_dir, local_files_only=True)
    pipe.load_components(names=["vae"], dtype=torch.bfloat16,
                         pretrained_model_name_or_path=modular_dir, local_files_only=True)
    vae = pipe.vae
    dtypes = {}
    for p in vae.parameters():
        dtypes[p.dtype] = dtypes.get(p.dtype, 0) + p.numel() * p.element_size()
    print("host bytes by dtype", {str(k): v / 1e9 for k, v in dtypes.items()}, flush=True)
    other = copy.deepcopy(vae)

    start = time.perf_counter()
    vae.to(device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"reference .to(cuda, bf16): {time.perf_counter() - start:.2f} s", flush=True)

    start = time.perf_counter()
    other.to(device="cuda:0")
    torch.cuda.synchronize()
    moved = time.perf_counter() - start
    other.to(device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"move then cast: {moved:.2f} + {time.perf_counter() - start - moved:.2f} s", flush=True)

    reference = dict(vae.state_dict())
    candidate = dict(other.state_dict())
    same = all(reference[k].dtype == candidate[k].dtype and torch.equal(reference[k].view(torch.uint8) if reference[k].dim() else reference[k], candidate[k].view(torch.uint8) if candidate[k].dim() else candidate[k])
               for k in reference)
    print("bitwise identical:", same, "tensors", len(reference), flush=True)


if __name__ == "__main__":
    main()
