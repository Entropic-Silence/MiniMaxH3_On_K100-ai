"""Diagnostic repeatability of the real H3 FP32 timestep MLP, both devices."""
import argparse
import json
import os
from pathlib import Path

import infinicore as ic
from infinicore.lib import _infinicore as ops
from infinicore.tensor import Tensor
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
from safetensors import safe_open
import torch
from diffusers.models.embeddings import get_timestep_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    timestep = torch.load(args.input_dump, weights_only=True)["inputs"]["timestep"]
    weights = {}
    for file in sorted((args.model_path / "transformer").glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith("time_embedder."):
                    weights[key.removeprefix("time_embedder.")] = source.get_tensor(key)
    assert len(weights) == 4 and all(w.dtype == torch.float32 for w in weights.values())
    report = {"diagnostic_only": True,
              "deterministic_blas": os.environ.get("H3_DETERMINISTIC_BLAS", "0"), "devices": []}
    for device in range(2):
        with torch.cuda.device(device):
            ic.set_device(ic.device("cuda", device))
            t = timestep.to("cuda")
            w = {k:v.to("cuda") for k,v in weights.items()}
            native_t = ic.from_torch(t)
            native_w = {k:ic.from_torch(v) for k,v in w.items()}
            torch.cuda.synchronize()
            refs = {"embedding": get_timestep_embedding(t, 256, flip_sin_to_cos=True,
                                                         downscale_freq_shift=0)}
            refs["linear_1"] = torch.nn.functional.linear(refs["embedding"], w["linear_1.weight"], w["linear_1.bias"])
            refs["activation"] = torch.nn.functional.silu(refs["linear_1"])
            refs["linear_2"] = torch.nn.functional.linear(refs["activation"], w["linear_2.weight"], w["linear_2.bias"])
            refs = {k:v.cpu() for k,v in refs.items()}
            values = {k:[] for k in refs}
            for _ in range(20):
                stages = {"embedding":ops.timestep_embedding(native_t._underlying)}
                stages["linear_1"] = ops.linear(stages["embedding"], native_w["linear_1.weight"]._underlying,
                                                native_w["linear_1.bias"]._underlying)
                stages["activation"] = ops.silu(stages["linear_1"])
                stages["linear_2"] = ops.linear(stages["activation"], native_w["linear_2.weight"]._underlying,
                                                native_w["linear_2.bias"]._underlying)
                for name, value in stages.items():
                    values[name].append(MiniMaxH3TransformerRunner._to_torch(Tensor(value), t.device).cpu())
            row = {"device":device, "stages":{}}
            for name, samples in values.items():
                row["stages"][name] = {
                    "all_20_equal":all(torch.equal(value,samples[0]) for value in samples),
                    "repeat_max_abs":max(float((value-samples[0]).abs().max()) for value in samples),
                    "reference_relative_l2":float((samples[0]-refs[name]).norm()/refs[name].norm())}
            report["devices"].append(row)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
