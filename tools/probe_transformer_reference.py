"""Diagnostic: compare full checkpoint DiT on identical captured or synthetic inputs."""
import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
import statistics
import time

import torch
from diffusers import MiniMaxH3Transformer3DModel
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner


def compare(actual, reference):
    diff = actual.float() - reference.float()
    return {"finite": bool(actual.isfinite().all()), "reference_finite": bool(reference.isfinite().all()),
            "actual_rms": float(actual.float().square().mean().sqrt()),
            "reference_rms": float(reference.float().square().mean().sqrt()),
            "max_abs": float(diff.abs().max()), "mean_abs": float(diff.abs().mean()),
            "relative_l2": float(diff.norm() / reference.float().norm().clamp_min(1e-12))}


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-tokens", type=int, default=0,
                        help="Diagnostic stage dump prefix, 0 stores all; computation always uses all tokens")
    parser.add_argument("--reference-cache", type=Path,
                        help="Diagnostic tensors only; never consumed by production generation")
    parser.add_argument("--input-dump", type=Path, help="Actual pipeline tensors from capture_first_dit.py")
    parser.add_argument("--repeat-native", action="store_true", help="Repeat identical input in the same engine")
    parser.add_argument("--compare-aten", action="store_true", help="Diagnostic existing attention backend ablation")
    parser.add_argument("--verify-fused", action="store_true", help="Require fused/unfused RMSNorm-AdaLN outputs to match")
    parser.add_argument("--benchmark-fusion-repeats", type=int, default=0,
                        help="Alternate warmed fused/unfused calls on identical full inputs; excludes weight loading")
    args = parser.parse_args()
    config = json.loads((args.model_path / "transformer/config.json").read_text())
    torch.manual_seed(42)
    inputs = {
        "hidden_states": torch.randn(1, 32, config["in_channels"] * 4, device="cuda:0", dtype=torch.float32),
        "audio_hidden_states": torch.randn(1, 16, config["audio_in_channels"], device="cuda:0", dtype=torch.float32),
        "encoder_hidden_states": torch.randn(1, 16, config["text_dim"], device="cuda:0", dtype=torch.bfloat16),
        "timestep": torch.tensor([.1, .3, .5], device="cuda:0"),
        "timestep_indices": torch.tensor([2]*32 + [1]*16 + [0]*16, device="cuda:0"),
        "token_tags": torch.tensor([1]*32 + [2]*16 + [0]*16, device="cuda:0"),
        "position_ids": torch.randn(64, 3, device="cuda:0"),
        "video_indices": torch.arange(32, device="cuda:0"),
        "audio_indices": torch.arange(32, 48, device="cuda:0"),
        "text_indices": torch.arange(48, 64, device="cuda:0"),
    }
    if args.input_dump:
        captured = torch.load(args.input_dump, weights_only=True)
        assert captured["diagnostic_only"]
        inputs = {key: value.to("cuda:0") for key, value in captured["inputs"].items()}
    report = {"diagnostic_only": True, "sequence_length": inputs["position_ids"].shape[0], "num_layers": config["num_layers"],
              "adaln_numerics": os.environ.get("H3_ADALN_NUMERICS", "exact_ref"),
              "pointwise_numerics": os.environ.get("H3_POINTWISE_NUMERICS", "baseline"),
              "fused_adaln": os.environ.get("H3_FUSED_ADALN", "0"),
              "deterministic_blas": os.environ.get("H3_DETERMINISTIC_BLAS", "0"),
              "rope_numerics": os.environ.get("H3_ROPE_NUMERICS", "baseline"),
              "attention": os.environ.get("INFINICORE_HYGON_MHA", "ninetoothed")}
    stages = {}
    if args.reference_cache and args.reference_cache.exists():
        cached = torch.load(args.reference_cache, weights_only=True)
        assert cached["config"] == config, "reference checkpoint config differs"
        assert cached.get("trace_tokens", 0) == args.trace_tokens
        if args.trace_dir:
            assert cached["stages"], "cache has no stage tensors"
        for key, value in inputs.items():
            assert torch.equal(value.cpu(), cached["inputs"][key]), f"cached input differs: {key}"
        inputs = {key: value.to("cuda:0") for key, value in cached["inputs"].items()}
        reference, stages = cached["reference"], cached["stages"]
        report["reference_cache"] = str(args.reference_cache)
    else:
        reference, stages, reference_info = load_reference(args, inputs)
        report.update(reference_info)
        if args.reference_cache:
            torch.save({"config": config, "inputs": {k:v.cpu() for k,v in inputs.items()},
                        "reference": reference, "stages": stages,
                        "trace_tokens": args.trace_tokens}, args.reference_cache)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    os.environ["H3_TEXT_ENCODER"] = "sequential"
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        os.environ["H3_TRACE_DIR"] = str(args.trace_dir)
        os.environ["H3_TRACE_TOKENS"] = str(args.trace_tokens)
    runner = MiniMaxH3TransformerRunner(args.model_path / "transformer", tp_device_ids=[0, 1])
    adapter = MiniMaxH3DiffusersAdapter(runner)
    with torch.inference_mode():
        video, audio = adapter(**inputs, return_dict=False)
    report["video"] = compare(video.cpu(), reference[0])
    report["audio"] = compare(audio.cpu(), reference[1])
    initial = (video.cpu(), audio.cpu())
    if args.trace_dir:
        report["stages"] = {}
        for name, ref in stages.items():
            file = args.trace_dir / f"{name}.bin"
            native_dtype = torch.float32 if name == "temb" else torch.bfloat16
            value = torch.frombuffer(bytearray(file.read_bytes()), dtype=native_dtype).reshape(ref.shape)
            report["stages"][name] = compare(value, ref.to(native_dtype))
    os.environ.pop("H3_TRACE_DIR", None)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.repeat_native:
        if args.trace_dir:
            repeat_dir = args.trace_dir / "repeat"
            repeat_dir.mkdir(exist_ok=True)
            os.environ["H3_TRACE_DIR"] = str(repeat_dir)
        with torch.inference_mode():
            repeat = adapter(**inputs, return_dict=False)
        report["repeat"] = {"video_equal":torch.equal(repeat[0].cpu(), initial[0]),
                            "audio_equal":torch.equal(repeat[1].cpu(), initial[1]),
                            "video":compare(repeat[0].cpu(), initial[0]),
                            "audio":compare(repeat[1].cpu(), initial[1])}
        if args.trace_dir:
            report["repeat"]["stages"] = {}
            for name, ref in stages.items():
                dtype = torch.float32 if name == "temb" else torch.bfloat16
                tensors = [torch.frombuffer(bytearray((directory / f"{name}.bin").read_bytes()),
                                            dtype=dtype).reshape(ref.shape)
                           for directory in (args.trace_dir, repeat_dir)]
                report["repeat"]["stages"][name] = compare(tensors[1], tensors[0])
        os.environ.pop("H3_TRACE_DIR", None)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.compare_aten:
        previous = os.environ.get("INFINICORE_HYGON_MHA")
        os.environ["INFINICORE_HYGON_MHA"] = "aten"
        try:
            with torch.inference_mode():
                baseline = adapter(**inputs, return_dict=False)
            report["aten"] = {"video":compare(baseline[0].cpu(),reference[0]),
                              "audio":compare(baseline[1].cpu(),reference[1])}
        finally:
            if previous is None:
                os.environ.pop("INFINICORE_HYGON_MHA", None)
            else:
                os.environ["INFINICORE_HYGON_MHA"] = previous
    if args.verify_fused:
        assert os.environ.get("H3_FUSED_ADALN", "0") == "0", "Start with unfused mode"
        os.environ["H3_FUSED_ADALN"] = "1"
        try:
            with torch.inference_mode():
                fused = adapter(**inputs, return_dict=False)
            report["fused"] = {"video_equal":torch.equal(fused[0].cpu(), initial[0]),
                               "audio_equal":torch.equal(fused[1].cpu(), initial[1]),
                               "video":compare(fused[0].cpu(),initial[0]),
                               "audio":compare(fused[1].cpu(),initial[1])}
        finally:
            os.environ["H3_FUSED_ADALN"] = "0"
    if args.benchmark_fusion_repeats:
        assert args.benchmark_fusion_repeats > 0
        assert os.environ.get("H3_FUSED_ADALN", "0") == "0", "Start with unfused mode"
        samples = {"0": [], "1": []}
        try:
            for iteration in range(args.benchmark_fusion_repeats + 1):
                for mode in ("0", "1"):
                    os.environ["H3_FUSED_ADALN"] = mode
                    for device in (0, 1):
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = adapter(**inputs, return_dict=False)
                    for device in (0, 1):
                        torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - started
                    # Compare after timing, against the same complete model input/output.
                    assert all(torch.equal(value.cpu(), expected) for value, expected in zip(result, initial))
                    if iteration:
                        samples[mode].append(seconds)
            report["fusion_benchmark"] = {"samples_seconds":samples,
                "median_seconds": {mode:statistics.median(values) for mode, values in samples.items()},
                "all_outputs_bitwise_equal":True,
                "scope":"Full DiT adapter on one captured input, two GPUs; excludes load and CPU comparison"}
        finally:
            os.environ["H3_FUSED_ADALN"] = "0"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    runner._residency.release()
    if args.verify_fused:
        assert report["fused"]["video_equal"] and report["fused"]["audio_equal"]


def load_reference(args, inputs):
    report, stages = {}, {}
    started = time.perf_counter()
    legacy = MiniMaxH3Transformer3DModel.from_pretrained(
        str(args.model_path / "transformer"), dtype=torch.bfloat16, device_map="balanced",
        max_memory={i:"54GiB" for i in (0, 1)}, low_cpu_mem_usage=True, local_files_only=True)
    legacy.eval().requires_grad_(False)
    official_path = Path(__file__).resolve().parents[1] / "InfiniLM/examples/minimax_h3.py"
    spec = importlib.util.spec_from_file_location("official_minimax_h3_probe", official_path)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    official.install_legacy_output_device_hooks(legacy)
    if args.trace_dir:
        def save(name):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    output = torch.cat(output, dim=-1)
                if args.trace_tokens > 0 and output.ndim == 3:
                    output = output[:, :args.trace_tokens]
                stages[name] = output.detach().cpu()
            return hook
        def save_input(name):
            def hook(module, inputs):
                value = inputs[0]
                if args.trace_tokens > 0 and value.ndim == 3:
                    value = value[:, :args.trace_tokens]
                stages[name] = value.detach().cpu()
            return hook
        for name, module in (("video_embeds", legacy.proj_in), ("audio_embeds", legacy.audio_proj_in),
                             ("context_embeds", legacy.context_embedder), ("text_embeds", legacy.token_refiner),
                             ("temb", legacy.time_embedder), ("norm_out", legacy.norm_out)):
            module.register_forward_hook(save(name))
        for i, block in enumerate(legacy.transformer_blocks):
            block.register_forward_hook(save(f"block{i}"))
        first = legacy.transformer_blocks[0]
        first.register_forward_pre_hook(save_input("packed"))
        for name, module in (("modulation", first.adaln_proj), ("norm1", first.norm1),
                             ("norm2", first.norm2), ("attention", first.attn), ("mlp", first.ff)):
            module.register_forward_hook(save("block0."+name))
        first.attn.register_forward_pre_hook(save_input("block0.modulated1"))
        first.ff.register_forward_pre_hook(save_input("block0.modulated2"))
    report["reference_load_seconds"] = time.perf_counter() - started
    report["reference_device_map"] = legacy.hf_device_map
    with torch.inference_mode():
        video, audio = legacy(**inputs, return_dict=False)
        reference = (video.cpu(), audio.cpu())
    del video, audio, legacy
    gc.collect()
    for index in (0, 1):
        with torch.cuda.device(index):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return reference, stages, report


if __name__ == "__main__":
    main()
