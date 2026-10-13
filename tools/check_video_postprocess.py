#!/usr/bin/env python3
"""Diagnostic: device-side PIL postprocessing (install_video_postprocess) vs the
released VideoProcessor.postprocess_video: identical pixels, and the time of each."""
import time

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor

from infinilm.multimodal.minimax_h3_vae_parallel import install_video_postprocess


def main():
    torch.manual_seed(0)
    video = torch.rand(1, 3, 124, 768, 768, device="cuda")
    # exact .5 / 255 boundaries and the clamp ends, where rounding could differ
    video[0, :, :4] = (torch.randint(0, 511, (3, 4, 768, 768), device="cuda").float() / 510)
    video[0, 0, 4] = 0.0
    video[0, 1, 4] = 1.0
    reference_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)
    fast_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)
    install_video_postprocess(fast_processor)
    results = {}
    for name, processor in (("released", reference_processor), ("device", fast_processor)):
        processor.postprocess_video(video[:, :, :8], output_type="pil")
        torch.cuda.synchronize()
        started = time.perf_counter()
        frames = processor.postprocess_video(video, output_type="pil")
        results[name] = (time.perf_counter() - started, frames)
    ref, new = results["released"][1], results["device"][1]
    same = len(ref) == len(new) and all(
        len(a) == len(b) and all(x.mode == y.mode and np.array_equal(np.asarray(x), np.asarray(y))
                                 for x, y in zip(a, b)) for a, b in zip(ref, new))
    print(f"released {results['released'][0]:.3f} s, device {results['device'][0]:.3f} s, "
          f"frames {len(new[0])}, identical pixels {same}")


if __name__ == "__main__":
    main()
