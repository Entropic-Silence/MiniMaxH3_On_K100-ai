#!/usr/bin/env python3
"""Read-only import and optional real H3 interface preflight; never installs packages."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    names = ["torch", "triton", "numpy", "transformers", "huggingface-hub", "diffusers",
             "peft", "mistral-common", "tokenizers", "safetensors", "click", "hf-xet", "tomli"]
    result = {"python": sys.executable, "versions": {}, "failures": []}
    for name in names:
        try:
            result["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][name] = None

    def check(name, action):
        try:
            action()
        except Exception as exc:
            result["failures"].append({"check": name, "type": type(exc).__name__, "message": str(exc)})

    for name in ["skimage", "av", "diffusers", "transformers"]:
        check(name, lambda name=name: importlib.import_module(name))

    def pinned_diffusers():
        import diffusers
        root = Path(diffusers.__file__).parent
        manifest = json.loads((Path(__file__).resolve().parents[1] /
                               "configs/diffusers_source_sha256.json").read_text())
        mismatches = [name for name, digest in manifest["files"].items()
                      if not (root / name).is_file() or
                      hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
        result["diffusers_source"] = {"path": str(root), "revision": manifest["revision"],
                                      "checked_files": len(manifest["files"]), "mismatches": mismatches}
        if mismatches:
            raise RuntimeError(f"Installed diffusers differs from pinned source: {mismatches[:5]}")
    check("pinned diffusers source", pinned_diffusers)

    def interfaces():
        from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
        from transformers import Qwen3VLForConditionalGeneration
        result["interfaces"] = [cls.__name__ for cls in
                                (MiniMaxH3Transformer3DModel, ModularPipeline, Qwen3VLForConditionalGeneration)]
    check("H3 imports", interfaces)

    if args.native:
        check("native runner", lambda: importlib.import_module("infinilm.multimodal.minimax_h3"))

    if args.model_path:
        def model_interfaces():
            from diffusers import MiniMaxH3Scheduler, ModularPipeline
            from transformers import AutoProcessor
            pipe = ModularPipeline.from_pretrained(str(args.model_path), local_files_only=True)
            processor = AutoProcessor.from_pretrained(str(args.model_path / "processor"), local_files_only=True)
            tokens = processor.tokenizer.encode("An astronaut walks on the moon.", add_special_tokens=False)
            token_types = processor.create_mm_token_type_ids([tokens])
            assert len(token_types) == 1 and len(token_types[0]) == len(tokens)
            result["model"] = {"path": str(args.model_path), "pipeline": type(pipe).__name__,
                               "processor": type(processor).__name__, "processor_test_tokens": len(tokens),
                               "scheduler": {}}
            for component in ("scheduler", "audio_scheduler"):
                scheduler = MiniMaxH3Scheduler.from_pretrained(str(args.model_path / component), local_files_only=True)
                scheduler.set_timesteps(50)
                nfe = len(scheduler.timesteps)
                if nfe != 49:
                    raise RuntimeError(f"Pinned H3 steps=50 expected NFE=49, got {nfe} ({component})")
                result["model"]["scheduler"][component] = {
                    "steps_argument": 50, "nfe": nfe,
                    "sigmas_first3": scheduler.sigmas[:3].tolist(),
                    "sigmas_last3": scheduler.sigmas[-3:].tolist(),
                    "timesteps": scheduler.timesteps.tolist(),
                }
        check("H3 checkpoint interfaces", model_interfaces)

    print(json.dumps(result, indent=2))
    if result["failures"]:
        print("Runtime preflight failed; no packages were changed. See docs/dependency-blocker.md.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
