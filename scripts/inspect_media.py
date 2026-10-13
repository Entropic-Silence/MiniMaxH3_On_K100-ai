#!/usr/bin/env python3
"""Decode a generated file and check media specifications, not model correctness."""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    report = {"file": str(args.video), "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
              "bytes": args.video.stat().st_size, "correctness_acceptance": "not_evaluated"}
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        video = {"width":stream.width, "height":stream.height, "fps":str(stream.average_rate),
                 "codec":stream.codec_context.name, "declared_frames":stream.frames,
                 "duration":float(stream.duration * stream.time_base) if stream.duration else None,
                 "decoded_frames":0, "frame_means":[], "pixel_min":255, "pixel_max":0}
        for index, frame in enumerate(container.decode(stream)):
            pixels = frame.to_ndarray(format="rgb24")
            video["decoded_frames"] += 1
            video["frame_means"].append(float(pixels.mean()))
            video["pixel_min"] = min(video["pixel_min"], int(pixels.min()))
            video["pixel_max"] = max(video["pixel_max"], int(pixels.max()))
            if index in (0, (args.frames - 1)//2, args.frames - 1):
                Image.fromarray(pixels).save(args.frames_dir / f"frame-{index:03d}.png")
        report["video"] = video
    with av.open(str(args.video)) as container:
        audio = None
        if container.streams.audio:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=stream.rate)
            chunks = []
            for frame in container.decode(stream):
                chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
            chunks.extend(f.to_ndarray() for f in resampler.resample(None))
            waveform = np.concatenate(chunks, axis=1) if chunks else np.empty((stream.codec_context.channels, 0))
            audio = {"codec":stream.codec_context.name, "rate":stream.rate,
                     "channels":stream.codec_context.channels, "decoded_samples":waveform.shape[1],
                     "duration":float(stream.duration*stream.time_base) if stream.duration else None,
                     "finite":bool(np.isfinite(waveform).all()),
                     "rms":float(np.sqrt(np.square(waveform.astype(np.float64)).mean())) if waveform.size else None,
                     "abs_peak":float(np.abs(waveform).max()) if waveform.size else None}
        report["audio"] = audio
    report["spec_pass"] = (video["width"] == args.width and video["height"] == args.height
                           and video["decoded_frames"] == args.frames and video["fps"] == str(args.fps)
                           and video["duration"] is not None
                           and abs(video["duration"] - args.frames / args.fps) < 1 / args.fps
                           and audio is not None and audio["rate"] == 32000 and audio["channels"] == 2
                           and audio["duration"] is not None
                           and abs(audio["duration"] - args.frames / args.fps) < .1
                           and audio["finite"] and audio["decoded_samples"] > 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:value for key,value in report.items() if key != "video"}, indent=2))
    if not report["spec_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
