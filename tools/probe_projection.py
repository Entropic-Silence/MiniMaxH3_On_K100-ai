"""Compare the real FP32 H3 input projection before and after its BF16 cast."""
import argparse
import json
from pathlib import Path

import infinicore as ic
from infinicore.lib import _infinicore as ops
from infinicore.tensor import Tensor
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",type=Path,required=True)
    parser.add_argument("--input-dump",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage-cache",type=Path)
    parser.add_argument("--native-stage-dir",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    inputs = torch.load(args.input_dump,weights_only=True)["inputs"]
    weights = {}
    for file in sorted((args.model_path/"transformer").glob("*.safetensors")):
        with safe_open(file,framework="pt",device="cpu") as source:
            for key in source.keys():
                if key in ("proj_in.weight","proj_in.bias"):
                    weights[key] = source.get_tensor(key)
    report = {"diagnostic_only":True,"allow_tf32":torch.backends.cuda.matmul.allow_tf32,"devices":[]}
    for device in range(2):
        with torch.cuda.device(device), torch.inference_mode():
            ic.set_device(ic.device("cuda",device))
            x = inputs["hidden_states"].to("cuda")
            w,b = weights["proj_in.weight"].to("cuda"), weights["proj_in.bias"].to("cuda")
            native = [ic.from_torch(t) for t in (x,w,b)]
            torch.cuda.synchronize()
            value = ops.linear(*(t._underlying for t in native))
            result = MiniMaxH3TransformerRunner._to_torch(Tensor(value),x.device)
            expected = torch.nn.functional.linear(x,w,b)
            row = {"device":device,"dtype":str(w.dtype),"shape":list(result.shape)}
            for name,a,r in (("fp32",result,expected),("bf16",result.to(torch.bfloat16),expected.to(torch.bfloat16))):
                delta = a.float()-r.float()
                row[name] = {"relative_l2":float(delta.norm()/r.float().norm()),
                             "max_abs":float(delta.abs().max()),"equal_fraction":float((a==r).float().mean())}
            report["devices"].append(row)
            if args.stage_cache:
                cache = torch.load(args.stage_cache,weights_only=True)
                assert torch.equal(inputs["hidden_states"],cache["inputs"]["hidden_states"])
                ref = cache["stages"]["video_embeds"].to(torch.bfloat16).float()
                direct = expected[:,:ref.shape[1]].cpu().to(torch.bfloat16).float()
                row["direct_vs_cached_reference"] = float((direct-ref).norm()/ref.norm())
                if args.native_stage_dir:
                    actual = torch.frombuffer(bytearray((args.native_stage_dir/"video_embeds.bin").read_bytes()),
                                               dtype=torch.bfloat16).reshape(ref.shape).float()
                    row["direct_vs_cached_native"] = float((direct-actual).norm()/direct.norm())
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
