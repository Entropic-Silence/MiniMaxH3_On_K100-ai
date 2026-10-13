#!/usr/bin/env python3
"""Diagnostic: time the Diffusers "remaining components" load of the H3 CLI.

Loads the same components the official entry loads after the text encoder
and DiT (VAE, audio VAE, processors, schedulers) and prints where the time
goes: from_pretrained per component, then the device moves.
"""
import argparse
import cProfile
import pstats
import time

import torch
from diffusers import ModularPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modular-dir", required=True)
    parser.add_argument("--workflow", default="t2va")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    pipe = ModularPipeline.from_pretrained(args.modular_dir, local_files_only=True)
    expected = {spec.name for spec in pipe.blocks.get_workflow(args.workflow).expected_components}
    names = [n for n, spec in pipe._component_specs.items()
             if spec.default_creation_method == "from_pretrained" and n in expected
             and not n.startswith(("text_encoder", "transformer"))]
    total = 0.0
    for name in names:
        start = time.perf_counter()
        runner = cProfile.Profile() if args.profile and name == "vae" else None
        if runner:
            runner.enable()
        pipe.load_components(names=[name], dtype=torch.bfloat16,
                             pretrained_model_name_or_path=args.modular_dir, local_files_only=True)
        if runner:
            runner.disable()
        seconds = time.perf_counter() - start
        total += seconds
        print(f"load {name}: {seconds:.2f} s", flush=True)
        if runner:
            pstats.Stats(runner).sort_stats("cumulative").print_stats(18)
    for name in ("vae", "audio_vae"):
        start = time.perf_counter()
        getattr(pipe, name).to(device="cuda:0", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print(f"{name}.to(cuda): {time.perf_counter() - start:.2f} s", flush=True)
    print(f"from_pretrained total {total:.2f} s")


if __name__ == "__main__":
    main()
