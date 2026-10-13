"""Reconstruct a stage-test input from an original captured input and prior-step diagnostic latents."""
import argparse
from pathlib import Path

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--first-input", type=Path, required=True)
    parser.add_argument("--previous-step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    first = torch.load(args.first_input, weights_only=True)
    previous = torch.load(args.previous_step, weights_only=True)
    assert first["diagnostic_only"] and previous["diagnostic_only"]
    inputs = first["inputs"]
    i = previous["step"] + 1
    schedulers = [MiniMaxH3Scheduler.from_pretrained(str(args.model_path/name), local_files_only=True)
                  for name in ("scheduler", "audio_scheduler")]
    for scheduler in schedulers:
        scheduler.set_timesteps(50)
    assert 0 <= i < len(schedulers[0].timesteps)
    vi, ai = inputs["video_indices"], inputs["audio_indices"]
    rows = inputs["timestep"][inputs["timestep_indices"]]
    ncv = int((rows[vi] > 0).sum())
    keyframe_aug = float(rows[vi[:ncv]].max()) if ncv else .999
    t, ta = (float(scheduler.timesteps[i]) for scheduler in schedulers)
    inputs["timestep"], inputs["timestep_indices"] = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        vi, ai, ncv, 0, inputs["text_indices"].numel(), t, ta, max(t,keyframe_aug), 1.)
    inputs["hidden_states"] = previous["latents"].unsqueeze(0)
    inputs["audio_hidden_states"] = previous["audio_latents"].unsqueeze(0)
    torch.save({"diagnostic_only":True, "step":i, "inputs":inputs}, args.output)
    print(f"Saved diagnostic input for step {i}: {args.output}")


if __name__ == "__main__":
    main()
