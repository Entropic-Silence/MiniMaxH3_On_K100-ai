"""Diagnostic only: isolate the unchanged H3 VAE from all native DiT kernels."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLMiniMaxH3
from skimage.metrics import structural_similarity


def stats(tensor):
    value = tensor.float()
    return {"shape": list(value.shape), "finite": bool(value.isfinite().all()),
            "mean": float(value.mean()), "std": float(value.std()), "min": float(value.min()), "max": float(value.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(args.image).convert("RGB").resize((768, 768)))
    report = {"diagnostic_only": True, "frames": 124, "modes": {}}
    for mode, dtype in (("official_cli_bf16", torch.bfloat16), ("checkpoint_fp32_diagnostic", torch.float32)):
        started = time.perf_counter()
        model = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model_path / "vae"), dtype=torch.bfloat16,
                                                     local_files_only=True)
        model.to(device="cuda:0", dtype=dtype).eval().requires_grad_(False)
        mean = torch.tensor([.485, .456, .406], device="cuda:0").view(1, 3, 1, 1, 1)
        std = torch.tensor([.229, .224, .225], device="cuda:0").view(1, 3, 1, 1, 1)
        pixels = torch.tensor(original.copy(), device="cuda:0").permute(2, 0, 1)[None, :, None]
        pixels = ((pixels.float() / 255 - mean) / std).repeat(1, 1, 124, 1, 1)
        with torch.inference_mode():
            latent = model.encode(pixels).latent_dist.sample(generator=torch.Generator().manual_seed(42)).to(torch.float16).float()
            item = {"latent": stats(latent), "tiling": model.use_tiling}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                decoded = model.decode(latent).sample
            item["decoded"] = stats(decoded)
            first = ((decoded[:, :, :1].float() * std + mean).clamp(0, 1)[0, :, 0] * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(first).save(args.output_dir / f"{mode}.png")
            item["first_frame_ssim"] = float(structural_similarity(original, first, channel_axis=2, data_range=255))
            item["seconds"] = time.perf_counter() - started
        report["modes"][mode] = item
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(mode, json.dumps(item), flush=True)
        del model, pixels, latent, decoded, mean, std
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
