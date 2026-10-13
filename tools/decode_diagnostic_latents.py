"""Decode complete-recurrence diagnostic latents with unchanged Diffusers blocks."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.video_processor import VideoProcessor
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep, MiniMaxH3VideoDecodeStep
from probe_denoise import stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-rows", type=int, default=576)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.dump, weights_only=True)
    assert data["diagnostic_only"] and data["step"] == 48
    model = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model_path / "vae"), dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.to(device="cuda:0", dtype=torch.bfloat16).eval().requires_grad_(False)
    components = SimpleNamespace(vae=model, patch_size=(1,2,2), vae_latent_channels=24, audio_channels=2,
                pixel_mean=(.485,.456,.406), pixel_std=(.229,.224,.225), _execution_device=torch.device("cuda:0"),
                video_processor=VideoProcessor(vae_scale_factor=16, do_normalize=False))
    state = PipelineState(values={"latents":data["latents"].to("cuda:0"),
            "audio_latents":data["audio_latents"].to("cuda:0"), "num_condition_video_rows":args.condition_rows,
            "num_condition_audio_rows":0, "num_latent_frames":37, "latent_height":48, "latent_width":48,
            "num_audio_latents":207, "output_type":"pt"})
    with torch.inference_mode():
        _,state = MiniMaxH3AfterDenoiseStep()(components,state)
        report = {"diagnostic_only":True, "packed_latents":stats(data["latents"]),
                  "unpacked_latents":stats(state.latents)}
        _,state = MiniMaxH3VideoDecodeStep()(components,state)
    frames = state.videos[0].cpu()
    report["decoded_shape"] = list(frames.shape)
    report["decoded"] = stats(frames)
    for i in (0,61,123):
        pixels = (frames[i].clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
        Image.fromarray(pixels).save(args.output_dir/f"frame-{i:03d}.png")
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
