#!/usr/bin/env python3
"""Exercise actual C++ MHA or HIP AOT on both cards and H3 sequence lengths."""
import argparse
import json
from pathlib import Path
import torch
from h3_attention import make_attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["aot", "infinicore"], default="aot")
    p.add_argument("--seqs", default="1024,8192,40000")
    p.add_argument("--heads", default="14,28,56")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--numerics", choices=["fp32_qk", "bf16_prescale", "fp32_qk_dotacc", "fp32_qk_split"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()) if args.config else dict(block_m=128, block_n=32, num_warps=4, num_stages=2)
    configured_numerics = config.pop("numerics", "fp32_qk")
    args.numerics = args.numerics or configured_numerics
    if args.backend == "aot":
        kernel = make_attention(caller="hip", output_dir=args.output.parent / "aot", numerics=args.numerics,
                                head_dim=args.head_dim, **config)
    else:
        import infinicore as ic
        def to_ic(t):
            # Upstream from_torch currently assumes contiguous storage. Preserve
            # the packed QKV and transposed output strides explicitly in this test.
            return ic.strided_from_blob(t.data_ptr(), list(t.shape), list(t.stride()),
                                        dtype=ic.bfloat16, device=ic.device("cuda", t.device.index))
    rows = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            if args.backend == "infinicore":
                ic.set_device(ic.device("cuda", device))
            for seq in map(int, args.seqs.split(",")):
                for heads in map(int, args.heads.split(",")):
                    torch.manual_seed(2026)
                    packed = torch.randn(1, seq, 3, heads, args.head_dim, dtype=torch.bfloat16, device="cuda")
                    q, k, v = (packed[:, :, i].transpose(1, 2) for i in range(3))
                    out = torch.empty_like(q)
                    def run():
                        if args.backend == "aot":
                            kernel(q, k, v, args.head_dim**-0.5, out)
                        else:
                            torch.cuda.synchronize()
                            ic.mha(*(to_ic(t.transpose(1, 2)) for t in (q,k,v)),
                                   scale=args.head_dim**-0.5, is_causal=False,
                                   out=to_ic(out.transpose(1, 2)))
                            ic.sync_stream()
                    run()
                    torch.cuda.synchronize()
                    first = out.clone()
                    run()
                    torch.cuda.synchronize()
                    deterministic = torch.equal(first, out)
                    ref = torch.nn.functional.scaled_dot_product_attention(q,k,v)
                    delta = (out.float() - ref.float()).abs()
                    indices = torch.linspace(0,seq-1,min(16,seq),device="cuda").long()
                    fp_logits = (q[:,:,indices].float() @ k.float().transpose(-2,-1)) * (args.head_dim**-0.5)
                    fp_ref = fp_logits.softmax(-1) @ v.float()
                    fp_delta = (out[:,:,indices].float() - fp_ref).abs()
                    row = {"backend": args.backend, "device":device,"seq":seq,"heads":heads,"head_dim":args.head_dim,
                           "arch": torch.cuda.get_device_properties(device).gcnArchName,
                           "numerics":args.numerics,"deterministic":deterministic,
                           "max_error":delta.max().item(),"mean_error":delta.mean().item(),
                           "fp32_sample_max_error":fp_delta.max().item(),"fp32_sample_mean_error":fp_delta.mean().item()}
                    row["sdpa_alignment_passed"] = row["max_error"] <= .002 and row["mean_error"] <= .0002
                    row["fp32_sample_passed"] = row["fp32_sample_max_error"] <= .002 and row["fp32_sample_mean_error"] <= .0002
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                    args.output.write_text(json.dumps(rows,indent=2)+"\n")
                    del packed,q,k,v,out,first,ref,delta,fp_logits,fp_ref,fp_delta
    if not all(r["deterministic"] and r["sdpa_alignment_passed"] and r["fp32_sample_passed"] for r in rows):
        raise SystemExit("Some configurations failed thresholds; see report")


if __name__ == "__main__":
    main()
