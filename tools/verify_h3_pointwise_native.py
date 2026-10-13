"""Exercise real InfiniCore C++ NineToothed dispatch against torch reference."""
import argparse
import json
from pathlib import Path

import infinicore as ic
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    results = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            ic.set_device(ic.device("cuda", device))
            def wrap(x):
                return ic.strided_from_blob(x.data_ptr(), list(x.shape), list(x.stride()),
                                            dtype=ic.bfloat16, device=ic.device("cuda", device))._underlying
            for operation, width in (("rms", 128), ("rms", 5376), ("swiglu", 7168), ("rmsmod", 5376)):
                for scale in (.01, 1., 100.):
                    packed = torch.randn(2, 129, width * 2 + 1, device="cuda", dtype=torch.bfloat16) * scale
                    x = packed[..., 1:width+1]
                    if operation in ("rms", "rmsmod"):
                        weight = torch.randn(width * 2, device="cuda", dtype=torch.bfloat16)[::2]
                        ref = torch.nn.functional.rms_norm(x, (width,), weight, eps=1e-5)
                        if operation == "rmsmod":
                            table = torch.randn(2, 129, width * 6, device="cuda", dtype=torch.bfloat16)
                            shift, modulation_scale = table[..., :width], table[..., width:2*width]
                            modulation_scale[:, :3].fill_(-1.)
                            ref = ref * (1 + modulation_scale) + shift
                            value = _infinicore.h3_rms_modulate(wrap(x), wrap(weight), wrap(modulation_scale), wrap(shift), 1e-5)
                        else:
                            value = _infinicore.h3_rms_norm(wrap(x), wrap(weight), 1e-5)
                    else:
                        gate = packed[..., width+1:]
                        ref = x * torch.nn.functional.silu(gate)
                        value = _infinicore.h3_swiglu(wrap(x), wrap(gate))
                    out = MiniMaxH3TransformerRunner._to_torch(Tensor(value), x.device)
                    equal = torch.equal(out, ref)
                    results.append({"device":device, "operation":operation, "width":width,
                                    "scale":scale, "equal":equal,
                                    "max_abs":float((out.float()-ref.float()).abs().max())})
            for seq, heads in ((17, 28), (257, 56), (2304, 28)):
                x = torch.randn(seq, heads, 385, device="cuda", dtype=torch.bfloat16)[..., 1:129]
                angles = torch.randn(seq, 97, device="cuda", dtype=torch.bfloat16)
                c, s = angles[:, 1:49], angles[:, 49:97]
                cosine, sine = torch.cat((c, c), -1)[:, None], torch.cat((s, s), -1)[:, None]
                rotated = torch.cat((-x[..., 48:96], x[..., :48]), -1)
                ref = torch.cat((x[..., :96] * cosine + rotated * sine, x[..., 96:]), -1)
                torch.cuda.synchronize(device)
                value = _infinicore.h3_rope(wrap(x), wrap(c), wrap(s))
                out = MiniMaxH3TransformerRunner._to_torch(Tensor(value), x.device)
                results.append({"device":device, "operation":"rope", "seq":seq, "heads":heads,
                                "equal":torch.equal(out, ref),
                                "max_abs":float((out.float()-ref.float()).abs().max())})
                cache = torch.cat((c, s), -1).contiguous()
                positions = torch.randperm(seq, device="cuda")
                cc, ss = c[positions], s[positions]
                ref = torch.cat((x[..., :96] * torch.cat((cc, cc), -1)[:, None]
                                 + rotated * torch.cat((ss, ss), -1)[:, None], x[..., 96:]), -1)
                torch.cuda.synchronize(device)
                ordered = _infinicore.embedding(ic.from_torch(positions)._underlying, wrap(cache))
                value = _infinicore.h3_rope(wrap(x), ordered.narrow(1, 0, 48), ordered.narrow(1, 48, 48))
                out = MiniMaxH3TransformerRunner._to_torch(Tensor(value), x.device)
                results.append({"device":device, "operation":"gather_rope", "seq":seq, "heads":heads,
                                "equal":torch.equal(out, ref),
                                "max_abs":float((out.float()-ref.float()).abs().max())})
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    assert all(row["equal"] for row in results)


if __name__ == "__main__":
    main()
