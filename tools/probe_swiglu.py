"""Diagnostic comparison of existing SwiGLU and NineToothed reference rounding."""
import argparse
import json
from pathlib import Path
import torch
import infinicore as ic
from infinicore.nn.functional import swiglu
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
from h3_pointwise import make_swiglu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    kernel = make_swiglu()
    report = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            ic.set_device(ic.device("cuda", device))
            for scale in (1., 4., 16.):
                packed = torch.randn(1, 129, 14336, device="cuda", dtype=torch.bfloat16) * scale
                up, gate = packed.chunk(2, dim=-1)
                ref = up * torch.nn.functional.silu(gate)
                out = torch.empty_like(up)
                kernel(up, gate, out)
                def wrap(tensor):
                    return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                                dtype=ic.bfloat16, device=ic.device("cuda", device))
                native = MiniMaxH3TransformerRunner._to_torch(swiglu(wrap(up), wrap(gate)), up.device)
                torch.cuda.synchronize()
                def metrics(value):
                    delta = value.float()-ref.float()
                    return {"max_abs":float(delta.abs().max()), "relative_l2":float(delta.norm()/ref.float().norm()),
                            "equal_fraction":float((value==ref).float().mean())}
                report.append({"device":device,"scale":scale,"native":metrics(native),"ninetoothed":metrics(out)})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
