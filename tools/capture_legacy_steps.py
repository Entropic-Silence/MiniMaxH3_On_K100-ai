"""Diagnostic only: record the legacy DiT's inputs and outputs at selected steps.

Runs the unchanged official CLI with `--legacy`, wrapping the Diffusers
transformer's forward so that the given (0-based) call indices are saved as
`<dir>/step_XX.pt` = {inputs, reference outputs}. Stops after the last one.
Pass the official CLI arguments after `--steps-to-save 0,12,24 --output-dir DIR`.
"""
import argparse
import functools
import inspect
from pathlib import Path
import runpy
import sys

import torch
from diffusers import MiniMaxH3Transformer3DModel


class CaptureComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--steps-to-save", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, official_args = parser.parse_known_args()
    wanted = sorted(int(s) for s in args.steps_to_save.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = MiniMaxH3Transformer3DModel.forward
    signature = inspect.signature(original)
    calls = {"n": 0}

    @functools.wraps(original)
    def capture(self, *values, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        output = original(self, *values, **kwargs)
        if index in wanted:
            bound = signature.bind(self, *values, **kwargs)
            inputs = {k: v.detach().cpu() for k, v in bound.arguments.items() if isinstance(v, torch.Tensor)}
            video, audio = (output.sample, output.audio_sample) if hasattr(output, "sample") else output[:2]
            torch.save({"diagnostic_only": True, "step": index, "inputs": inputs,
                        "reference": {"video": video.detach().float().cpu(),
                                      "audio": audio.detach().float().cpu()}},
                       args.output_dir / f"step_{index:02d}.pt")
            print(f"[diagnostic] saved legacy DiT step {index}", flush=True)
            if index == wanted[-1]:
                raise CaptureComplete()
        return output

    MiniMaxH3Transformer3DModel.forward = capture
    official = Path(__file__).resolve().parents[1] / "InfiniLM/examples/minimax_h3.py"
    sys.argv = [str(official), *official_args]
    try:
        runpy.run_path(str(official), run_name="__main__")
    except CaptureComplete:
        pass
    except Exception as exc:  # diffusers wraps block errors; accept our own stop
        if "CaptureComplete" not in repr(exc) and not isinstance(exc.__cause__, CaptureComplete):
            raise
    finally:
        MiniMaxH3Transformer3DModel.forward = original


if __name__ == "__main__":
    main()
