#!/usr/bin/env python3
"""Diagnostic: run the unified entry (InfiniLM/examples/minimax_h3.py, loaded
unchanged) and record a hash of the inputs and outputs of every timed
component call of the generation (condition encoder, VAE encodes, each DiT
forward, VAE decodes), one JSON line per call. Two runs of the same command
can then be compared call by call to find where they first differ
(tools/compare_traces.py). The hashing synchronizes the devices and copies
tensors to the host, so timings of a traced run are not representative.
Never used by generation.

usage: trace_pipeline.py OUT.jsonl <minimax_h3.py arguments...>
"""
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ENTRY = Path(__file__).resolve().parents[1] / "InfiniLM" / "examples" / "minimax_h3.py"


def digest(obj, depth=0):
    if torch.is_tensor(obj):
        tensor = obj.detach()
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        tensor = tensor.contiguous().cpu()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.view(torch.int16)
        return f"{list(obj.shape)}:{str(obj.dtype)[6:]}:" + hashlib.sha1(tensor.numpy().tobytes()).hexdigest()[:16]
    if depth > 3:
        return None
    if isinstance(obj, (list, tuple)):
        return [digest(item, depth + 1) for item in obj]
    if isinstance(obj, dict):
        return {str(key): digest(value, depth + 1) for key, value in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        found = {name: digest(value, depth + 1) for name, value in vars(obj).items()
                 if torch.is_tensor(value) or hasattr(value, "__dict__") or isinstance(value, (list, tuple, dict))}
        return {name: value for name, value in found.items() if value not in (None, [], {})} or None
    return None


def main():
    out_path = Path(sys.argv[1])
    sys.argv = [str(ENTRY)] + sys.argv[2:]
    spec = importlib.util.spec_from_file_location("h3_entry", ENTRY)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry  # dataclasses look their module up there
    spec.loader.exec_module(entry)
    out = out_path.open("w")
    counts = {}

    def wrap(owner, method_name, name):
        original = getattr(owner, method_name)

        @functools.wraps(original)
        def traced(*args, **kwargs):
            result = original(*args, **kwargs)
            index = counts.get(name, 0)
            counts[name] = index + 1
            record = {"call": name, "index": index, "args": digest(list(args)),
                      "kwargs": digest(kwargs), "output": digest(result)}
            out.write(json.dumps(record) + "\n")
            out.flush()
            return result

        setattr(owner, method_name, traced)

    install = entry.install_component_profilers

    def install_traced(report, *, text_encoder, transformer, vae, audio_vae):
        profiler = install(report, text_encoder=text_encoder, transformer=transformer,
                           vae=vae, audio_vae=audio_vae)
        wrap(text_encoder.model, "forward", "condition_encoder")
        wrap(transformer, "forward", "transformer")
        wrap(vae, "encode", "video_vae_encode")
        wrap(audio_vae, "encode", "audio_vae_encode")
        wrap(vae, "decode", "video_vae_decode")
        wrap(audio_vae, "decode", "audio_vae_decode")
        return profiler

    entry.install_component_profilers = install_traced
    try:
        entry.main()
    finally:
        out.close()


if __name__ == "__main__":
    main()
