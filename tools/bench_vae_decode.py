#!/usr/bin/env python3
"""Diagnostic: time the H3 video VAE decode (the pipeline's tiled decode with
the tile-parallel scheduler) on random latents of a real shape, optionally
saving / comparing the decoded video. Never used by generation."""
import argparse
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLMiniMaxH3
from infinilm.multimodal.minimax_h3_vae_parallel import install_parallel_vae_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--latent", default="37,48,48", help="frames,height,width of the latent")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--cudnn-benchmark", action="store_true",
                        help="torch.backends.cudnn.benchmark (MIOpen find per new shape)")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    devices = [int(d) for d in args.gpus.split(",")]
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model_path / "vae", local_files_only=True)
    vae.to(f"cuda:{devices[0]}")
    vae.to(dtype=torch.bfloat16).requires_grad_(False).eval()
    install_parallel_vae_decode(vae, devices)
    frames, height, width = map(int, args.latent.split(","))
    generator = torch.Generator("cpu").manual_seed(0)
    z = torch.randn(1, 24, frames, height, width, generator=generator).to(f"cuda:{devices[0]}", torch.bfloat16)
    times = []
    with torch.inference_mode():
        for i in range(args.repeats + 1):
            for d in devices:
                torch.cuda.synchronize(d)
            start = time.perf_counter()
            # The pipeline decodes under fp16 autocast (decoders.py).
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                video = vae.decode(z, return_dict=False)[0]
            for d in devices:
                torch.cuda.synchronize(d)
            if i:
                times.append(time.perf_counter() - start)
    print(f"decode {tuple(video.shape)}: {[round(t, 3) for t in times]} s", flush=True)
    if args.save:
        torch.save(video.cpu(), args.save)
    if args.compare:
        ref = torch.load(args.compare)
        out = video.cpu()
        rel = float((out.float() - ref.float()).norm() / ref.float().norm())
        print(f"bitwise {torch.equal(out, ref)} rel_l2 {rel:.3e} max_abs {float((out.float() - ref.float()).abs().max()):.3e}")


if __name__ == "__main__":
    main()
