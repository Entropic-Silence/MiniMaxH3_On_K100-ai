"""Check actual native Qwen outputs and memory release across sequential reloads."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoProcessor
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder


def used_memory(devices):
    return [torch.cuda.mem_get_info(index)[1] - torch.cuda.mem_get_info(index)[0] for index in devices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--reference-transformers", action="store_true",
                        help="Also compare against the full released Transformers Qwen stack")
    args = parser.parse_args()
    devices = [int(x) for x in args.gpus.split(",")]
    processor = AutoProcessor.from_pretrained(str(args.model_path / "processor"), local_files_only=True)
    prompts = ["A dog falls asleep", "A cat slowly turns its head.", "A dog falls asleep"]
    report = {"prompts": prompts, "devices": devices, "modes": {}}
    legacy_outputs = []
    if args.reference_transformers:
        from transformers import Qwen3VLForConditionalGeneration
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
        started = time.perf_counter()
        legacy = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model_path / "text_encoder"), dtype=torch.bfloat16,
            device_map="balanced_low_0", low_cpu_mem_usage=True, local_files_only=True,
            max_memory={index: f"{int(torch.cuda.get_device_properties(index).total_memory / 1024**3 * .85)}GiB"
                        for index in devices})
        legacy.eval().requires_grad_(False)
        report["legacy"] = {"create_seconds": time.perf_counter() - started,
                            "device_map": legacy.hf_device_map, "num_layers": legacy.config.text_config.num_hidden_layers}
        with torch.inference_mode():
            for prompt in prompts:
                ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
                legacy_outputs.append(get_qwen3vl_prompt_embeds(
                    legacy, processor, ids, device=torch.device("cuda", devices[0]), dtype=torch.bfloat16).cpu())
        del legacy
        gc.collect()
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    references = []
    for mode in ("tp_all", "sequential"):
        os.environ["H3_TEXT_ENCODER"] = mode
        before = used_memory(devices)
        started = time.perf_counter()
        encoder = MiniMaxH3ConditionEncoder(args.model_path / "text_encoder", tp_device_ids=devices)
        mode_report = {"create_seconds": time.perf_counter() - started, "before_bytes": before,
                       "after_create_bytes": used_memory(devices), "cases": []}
        report["modes"][mode] = mode_report
        for i, prompt in enumerate(prompts):
            inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs["mm_token_type_ids"] = torch.tensor(processor.create_mm_token_type_ids(inputs["input_ids"].tolist()))
            started = time.perf_counter()
            with torch.inference_mode():
                actual = encoder.encode(inputs).cpu()
            item = {"seconds": time.perf_counter() - started, "shape": list(actual.shape),
                    "after_encode_bytes": used_memory(devices), "engine_resident": encoder.engine is not None,
                    "finite": bool(actual.isfinite().all())}
            if mode == "tp_all":
                references.append(actual)
                if legacy_outputs:
                    diff = actual.float() - legacy_outputs[i].float()
                    item["legacy_relative_l2"] = float(diff.norm() / legacy_outputs[i].float().norm().clamp_min(1e-12))
                    item["legacy_max_abs"] = float(diff.abs().max())
                    item["legacy_pass"] = item["legacy_relative_l2"] < 1e-2
            else:
                diff = actual.float() - references[i].float()
                item.update(bitwise_equal=torch.equal(actual, references[i]), max_abs=float(diff.abs().max()),
                            relative_l2=float(diff.norm() / references[i].float().norm().clamp_min(1e-12)))
                assert item["relative_l2"] < 1e-3, item
                assert not item["engine_resident"], item
            assert item["finite"], item
            mode_report["cases"].append(item)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        encoder._residency.release()
        mode_report["after_release_bytes"] = used_memory(devices)
        del encoder
    report["resident_repeat_equal"] = torch.equal(references[0], references[2])
    report["legacy_pass"] = all(item.get("legacy_pass", True) for item in report["modes"]["tp_all"]["cases"]) if legacy_outputs else None
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["legacy_pass"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
