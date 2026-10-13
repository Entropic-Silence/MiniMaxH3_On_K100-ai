#!/usr/bin/env python3
"""Call the official comparison unchanged, then add stream specifications."""
import argparse
import math
import json
from itertools import zip_longest
from pathlib import Path
import av
import numpy as np
from skimage.metrics import structural_similarity
from run_case import official_compare


def specifications(path):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0] if container.streams.audio else None
        return {"width":video.width,"height":video.height,"video_codec":video.codec_context.name,
                "fps":str(video.average_rate),"audio_codec":audio.codec_context.name if audio else None,
                "sample_rate":audio.rate if audio else None,
                "channels":audio.codec_context.channels if audio else None}


def image_metrics(left, right, official):
    squared_error, elements, first_ssim = 0.0, 0, None
    comparable = True
    for index, (a, b) in enumerate(zip_longest(official.decoded_video_frames(left),
                                              official.decoded_video_frames(right))):
        if a is None or b is None:
            comparable = False
            continue
        a, b = a.to_ndarray(format="rgb24"), b.to_ndarray(format="rgb24")
        if a.shape != b.shape:
            comparable = False
            continue
        if index == 0:
            first_ssim = float(structural_similarity(official.prepare_ssim_frame(a),
                               official.prepare_ssim_frame(b), data_range=255)) * 100
        difference = a.astype(np.float64) - b.astype(np.float64)
        squared_error += float(np.square(difference).sum())
        elements += difference.size
    mse = squared_error / elements if elements and comparable else None
    psnr = None if mse is None else ("Infinity" if mse == 0 else 10 * math.log10(255**2 / mse))
    return {"first_frame_ssim_percent": first_ssim, "rgb_mse": mse, "psnr_db": psnr}


def decode_audio(path):
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None, None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=stream.rate)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray() for f in resampler.resample(None))
        return (np.concatenate(chunks, axis=1) if chunks else
                np.empty((stream.codec_context.channels, 0), dtype=np.float32)), stream.rate


def logmel(waveform, sample_rate):
    # Evaluation only: HTK triangular mel filters, power STFT, natural log.
    nfft, hop, bins = 1024, 256, 80
    waveform = np.pad(waveform, ((0, 0), (nfft // 2, nfft // 2)))
    windows = np.lib.stride_tricks.sliding_window_view(waveform, nfft, axis=-1)[:, ::hop]
    power = np.abs(np.fft.rfft(windows * np.hanning(nfft), axis=-1)) ** 2
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(hz_to_mel(20), hz_to_mel(sample_rate / 2), bins + 2) / 2595) - 1)
    frequencies = np.fft.rfftfreq(nfft, 1 / sample_rate)
    rising = (frequencies[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    falling = (edges[2:, None] - frequencies[None]) / (edges[2:, None] - edges[1:-1, None])
    filters = np.maximum(0, np.minimum(rising, falling))
    return np.log(np.maximum(power @ filters.T, 1e-10))


def audio_metrics(left, right):
    a, rate_a = decode_audio(left)
    b, rate_b = decode_audio(right)
    result = {"reference_present": a is not None, "candidate_present": b is not None,
              "waveform_correlation": None, "log_mel_l1": None}
    if a is None or b is None:
        result["status"] = "both_absent" if a is None and b is None else "audio_presence_mismatch"
        return result
    result.update(reference_samples=a.shape[1], candidate_samples=b.shape[1],
                  sample_count_matches=a.shape[1] == b.shape[1])
    if rate_a != rate_b or a.shape[0] != b.shape[0]:
        result["status"] = "audio_format_mismatch"
        return result
    n = min(a.shape[1], b.shape[1])
    if not n:
        result["status"] = "empty_audio"
        return result
    a, b = a[:, :n].astype(np.float64), b[:, :n].astype(np.float64)
    exact = np.array_equal(a, b)
    ca, cb = a.ravel() - a.mean(), b.ravel() - b.mean()
    denominator = np.linalg.norm(ca) * np.linalg.norm(cb)
    result.update(status="ok", compared_samples=n, exact_overlap=exact,
                  waveform_correlation=1.0 if exact else (float(ca @ cb / denominator) if denominator else None),
                  waveform_mad=float(np.abs(a - b).mean()),
                  log_mel_l1=float(np.abs(logmel(a, rate_a) - logmel(b, rate_b)).mean()),
                  log_mel_definition="80 HTK bins, 20Hz..Nyquist, FFT1024, hop256, Hann, power, ln floor1e-10")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("reference",type=Path)
    p.add_argument("candidate",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    official=official_compare()
    comparison=official.compare_video_streams(args.reference,args.candidate)
    left,right=specifications(args.reference),specifications(args.candidate)
    result={"official":comparison,"reference_specs":left,"candidate_specs":right,
            "spec_matches":{key:left[key]==right[key] for key in left},
            "additional_video_metrics":image_metrics(args.reference,args.candidate,official),
            "audio_metrics":audio_metrics(args.reference,args.candidate)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
