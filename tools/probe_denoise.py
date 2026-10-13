"""Diagnostic full 49-step DiT recurrence from captured real inputs.

Uses the unchanged Diffusers denoiser/scheduler blocks and complete checkpoint.
Captured conditioning is not a legacy reference; no video or contest score is produced.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep


def stats(x):
    x = x.float()
    return {"finite":bool(x.isfinite().all()), "mean":float(x.mean()), "std":float(x.std()),
            "min":float(x.min()), "max":float(x.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    captured = torch.load(args.input_dump, weights_only=True)
    assert captured["diagnostic_only"]
    inputs = {key:value.to("cuda:0") for key,value in captured["inputs"].items()}
    video_indices, audio_indices = inputs["video_indices"], inputs["audio_indices"]
    initial_rows = inputs["timestep"][inputs["timestep_indices"]]
    video_condition = initial_rows[video_indices] > 0
    ncv = int(video_condition.sum())
    assert video_condition[:ncv].all() and not video_condition[ncv:].any()
    assert not (initial_rows[audio_indices] > 0).any(), "This probe expects FL2VA without reference audio"
    keyframe_aug = float(initial_rows[video_indices[:ncv]].max()) if ncv else .999
    scheduler = MiniMaxH3Scheduler.from_pretrained(str(args.model_path / "scheduler"), local_files_only=True)
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(str(args.model_path / "audio_scheduler"), local_files_only=True)
    scheduler.set_timesteps(50, device="cuda:0")
    audio_scheduler.set_timesteps(50, device="cuda:0")
    assert len(scheduler.timesteps) == len(audio_scheduler.timesteps) == 49
    plan = [tuple(x.to("cuda:0") for x in MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices.cpu(), audio_indices.cpu(), ncv, 0, inputs["text_indices"].numel(),
            float(t), float(ta), max(float(t), keyframe_aug), 1.))
            for t,ta in zip(scheduler.timesteps, audio_scheduler.timesteps)]
    assert torch.equal(plan[0][0], inputs["timestep"])
    assert torch.equal(plan[0][1], inputs["timestep_indices"])
    runner = None
    if args.backend == "reference":
        model = MiniMaxH3Transformer3DModel.from_pretrained(str(args.model_path / "transformer"),
                dtype=torch.bfloat16, device_map="balanced", max_memory={0:"54GiB",1:"54GiB"},
                low_cpu_mem_usage=True, local_files_only=True).eval().requires_grad_(False)
        official_path = Path(__file__).resolve().parents[1] / "InfiniLM/examples/minimax_h3.py"
        spec = importlib.util.spec_from_file_location("official_h3_diagnostic", official_path)
        official = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = official
        spec.loader.exec_module(official)
        official.install_legacy_output_device_hooks(model)
    else:
        from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner, MiniMaxH3DiffusersAdapter
        os.environ["H3_TEXT_ENCODER"] = "sequential"
        runner = MiniMaxH3TransformerRunner(args.model_path / "transformer", tp_device_ids=[0,1])
        model = MiniMaxH3DiffusersAdapter(runner)
    components = SimpleNamespace(transformer=model, scheduler=scheduler, audio_scheduler=audio_scheduler)
    state = SimpleNamespace(latents=inputs["hidden_states"][0].clone(),
            audio_latents=inputs["audio_hidden_states"][0].clone(), prompt_embeds=inputs["encoder_hidden_states"],
            row_timestep_plan=plan, attention_kwargs=None,
            denoiser_input_fields={k:inputs[k] for k in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")},
            num_condition_video_rows=ncv, num_condition_audio_rows=0, audio_timesteps=audio_scheduler.timesteps)
    denoiser, step = MiniMaxH3LoopDenoiser(), MiniMaxH3LoopSchedulerStep()
    report = {"diagnostic_only":True, "backend":args.backend, "steps":50, "nfe":49,
              "num_condition_video_rows":ncv, "keyframe_noise_aug":keyframe_aug, "trajectory":[],
              "input_sha256":hashlib.sha256(args.input_dump.read_bytes()).hexdigest(),
              "environment":{key:os.environ.get(key) for key in (
                  "H3_POINTWISE_NUMERICS", "H3_ADALN_NUMERICS", "H3_FUSED_ADALN",
                  "H3_DETERMINISTIC_BLAS", "H3_ROPE_NUMERICS", "INFINICORE_HYGON_MHA")},
              "source_pins":json.loads((Path(__file__).resolve().parents[1] / "versions.json").read_text())}
    with torch.inference_mode():
        for i,t in enumerate(scheduler.timesteps):
            components,state = denoiser(components,state,i,t)
            item = {"step":i, "t":float(t), "video_in":stats(state.latents[ncv:]),
                    "video_pred":stats(state.noise_pred[0,ncv:]), "audio_pred":stats(state.audio_noise_pred)}
            components,state = step(components,state,i,t)
            item["video_out"] = stats(state.latents[ncv:])
            report["trajectory"].append(item)
            print(json.dumps(item), flush=True)
            if i in (0, 23, 48):
                torch.save({"diagnostic_only":True, "step":i, "latents":state.latents.cpu(),
                            "audio_latents":state.audio_latents.cpu()}, args.output_dir / f"step-{i:02d}.pt")
            (args.output_dir / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    if runner:
        runner._residency.release()
    del model, components, state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
