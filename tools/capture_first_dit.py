"""Diagnostic only: capture the official pipeline's first DiT call, then stop.

This is a stage-level numerical test, never a generated video or timing result.
Pass the unchanged official CLI arguments after --output-dump FILE.
"""
import argparse
import functools
import inspect
from pathlib import Path
import runpy
import sys

import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter


class CaptureComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dump", type=Path, required=True)
    args, official_args = parser.parse_known_args()
    original = MiniMaxH3DiffusersAdapter.forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def capture(self, *values, **kwargs):
        bound = signature.bind(self, *values, **kwargs)
        inputs = {key: value.detach().cpu() for key, value in bound.arguments.items()
                  if isinstance(value, torch.Tensor)}
        output = original(self, *values, **kwargs)
        video, audio = (output.sample, output.audio_sample) if hasattr(output, "sample") else output
        torch.save({"diagnostic_only": True, "inputs": inputs,
                    "native": (video.detach().cpu(), audio.detach().cpu())}, args.output_dump)
        print(f"[diagnostic] Captured first DiT inputs/output to {args.output_dump}; no video or benchmark produced.", flush=True)
        raise CaptureComplete()

    MiniMaxH3DiffusersAdapter.forward = capture
    official = Path(__file__).resolve().parents[1] / "InfiniLM/examples/minimax_h3.py"
    sys.argv = [str(official), *official_args]
    try:
        runpy.run_path(str(official), run_name="__main__")
    except CaptureComplete:
        pass
    finally:
        MiniMaxH3DiffusersAdapter.forward = original


if __name__ == "__main__":
    main()
