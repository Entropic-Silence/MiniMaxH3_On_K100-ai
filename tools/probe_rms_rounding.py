"""Measure the installed torch RMSNorm rounding; diagnostic inputs only."""
import argparse
import json
from pathlib import Path

import torch
from h3_pointwise import make_rms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    report = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            for dim in (128, 5376):
                kernel = make_rms(width=dim)
                weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
                for scale in (.01, 1., 100., 10000.):
                    x = torch.randn(129, dim, device="cuda", dtype=torch.bfloat16) * scale
                    eps = 1e-5
                    reference = torch.nn.functional.rms_norm(x, (dim,), weight, eps=eps)
                    eps_bf16 = torch.tensor(eps, device=x.device, dtype=x.dtype)
                    staged = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps_bf16) * weight
                    promoted = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
                                * weight.float()).to(x.dtype)
                    candidate = torch.empty_like(x)
                    kernel(x.unsqueeze(0), weight, eps, candidate.unsqueeze(0))
                    torch.cuda.synchronize()
                    def metrics(value):
                        difference = value.float() - reference.float()
                        return {"equal": bool(torch.equal(value, reference)),
                                "max_abs": float(difference.abs().max()),
                                "relative_l2": float(difference.norm() / reference.float().norm())}
                    report.append({"device": device, "dim": dim, "scale": scale,
                                   "staged_bf16": metrics(staged), "promoted_fp32": metrics(promoted),
                                   "ninetoothed": metrics(candidate)})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
