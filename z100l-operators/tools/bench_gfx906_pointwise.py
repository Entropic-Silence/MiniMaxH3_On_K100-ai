#!/usr/bin/env python3
"""Reuse H3 NineToothed kernels; qualify and tune on the actual contest DCU.
PyTorch is used only as a diagnostic reference. No handwritten device code.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_h3_ops as bench
import torch

MAKERS = {
    'swiglu': 'make_swiglu',
    'gated_residual': 'make_gated_residual',
    'qk_norm_rope': 'make_qk_norm_rope',
    'rope': 'make_rope', 'rope_cache': 'make_rope_cache',
    'rms': 'make_rms', 'rms_modulate': 'make_rms_modulate',
    'gated_residual_rms_modulate': 'make_gated_residual_rms_modulate',
    'attention': 'make_attention',
}
CANDIDATES = {
    'swiglu': [dict(rows=r, cols=c, num_warps=w) for r,c,w in
               [(1,256,4),(1,512,8),(1,1024,16),(1,128,2),(1,64,1),(16,256,4),(1,1024,4),(1,512,4),(1,2048,4),(2,1024,4),(4,512,4),(8,256,4),(1,1024,2),(1,2048,8)]],
    'gated_residual': [dict(block=[r,c],num_warps=w) for r,c,w in
                       [(4,512,8),(1,1024,4),(4,512,4),(8,256,4),(1,2048,4)]],
    'qk_norm_rope': [dict(rows=r,num_warps=w,block_tokens=t) for r,w,t in
                     [(8,1,1),(8,2,1),(8,4,1),(4,1,1),(16,2,1),(8,2,2)]],
}

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seq',type=int,default=8192)
p.add_argument('--heads',type=int,default=14)
p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
p.add_argument('--seed',type=int,default=0)
p.add_argument('--repeats',type=int,default=30)
p.add_argument('--samples',type=int,default=5)
p.add_argument('--ops',default='qk_norm_rope,rope,rope_cache,rms,rms_modulate,gated_residual,gated_residual_rms_modulate,swiglu')
p.add_argument('--caller',choices=['torch','hip'],default='hip')
p.add_argument('--aot-dir',type=Path,default=Path('/tmp/h3-gfx906-aot'))
p.add_argument('--tune',action='store_true')
p.add_argument('--broadcast-modulation',action='store_true')
p.add_argument('--config',type=Path)
p.add_argument('--json',type=Path,required=True)
a=p.parse_args()
cfg=type('Config',(),{})()
cfg.seq=a.seq;cfg.heads=a.heads;cfg.device='cuda:0';cfg.broadcast_modulation=a.broadcast_modulation
cfg.torch_dtype,cfg.nt_dtype=bench.DTYPES[a.dtype]
cfg.attention_config={'block_m':64,'block_n':32,'num_warps':4,'num_stages':1}
config=json.loads(a.config.read_text()) if a.config else {}
props=torch.cuda.get_device_properties(0)
import triton
results=[]
for name in a.ops.split(','):
    maker_name=MAKERS.get(name)
    original=getattr(bench,maker_name) if maker_name else None
    candidates=CANDIDATES.get(name,[{}]) if a.tune else [config.get(name,{})]
    for options in candidates:
        row={'operator':name,'options':options}
        print('BEGIN',name,a.caller,options,flush=True)
        try:
            if original:
                maker_options=dict(options)
                if a.caller=='hip':
                    digest=hashlib.sha256(json.dumps(options,sort_keys=True).encode()).hexdigest()[:12]
                    maker_options.update(caller='hip',archs=[props.gcnArchName.split(':')[0]],
                        output_dir=a.aot_dir/a.dtype/name/digest)
                factory = functools.partial(original, **maker_options)
                if a.caller == 'hip' and name == 'qk_norm_rope':
                    # HIP AOT exports the twelve-view ABI used by C++; the
                    # eager maker normally wraps this as six Python arguments.
                    def qk_factory(**kwargs):
                        raw = factory(**kwargs)
                        return lambda x,w,e,c,s,o: raw(x,x,x,w,w,w,e,c,s,o,o,o)
                    setattr(bench, maker_name, qk_factory)
                else:
                    setattr(bench,maker_name,factory)
            torch.manual_seed(a.seed)
            with torch.inference_mode():
                out,ref,run,baseline,flops=bench.OPERATORS[name](cfg)
                torch.cuda.synchronize()
                diff=(out.float()-ref.float())
                row.update(bitwise_equal=torch.equal(out.contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8)),
                    max_abs_difference=diff.abs().max().item(),
                    relative_l2=(diff.norm()/ref.float().norm().clamp_min(1e-30)).item(),
                    finite=bool(torch.isfinite(out).all()))
                km=[bench.timed(run,a.repeats) for _ in range(a.samples)]
                bm=[bench.timed(baseline,a.repeats) for _ in range(a.samples)]
                row.update(kernel_ms=statistics.median(km),eager_ms=statistics.median(bm),
                           kernel_samples_ms=km,eager_samples_ms=bm)
                row['speedup']=row['eager_ms']/row['kernel_ms']
                # Tight diagnostic threshold; non-bitwise results stay marked.
                atol,rtol=(0.001,0.001) if a.dtype=='fp16' else (0.01,0.01)
                row['atol']=atol;row['rtol']=rtol
                row['allclose']=torch.allclose(out,ref,atol=atol,rtol=rtol)
                row['correct']=row['finite'] and row['allclose'] and row['relative_l2'] < (0.001 if a.dtype=='fp16' else 0.005)
        except Exception as e:
            row.update(error=f'{type(e).__name__}: {e}',correct=False)
        finally:
            if original:setattr(bench,maker_name,original)
        results.append(row)
        print(json.dumps(row),flush=True)
        a.json.parent.mkdir(parents=True,exist_ok=True)
        a.json.write_text(json.dumps({'environment':{'device':props.name,'arch':props.gcnArchName,
            'torch':torch.__version__,'hip':torch.version.hip,'triton':triton.__version__,
            'portable_bf16':os.environ.get('H3_GFX906_PORTABLE_BF16')=='1','caller':a.caller,'broadcast_modulation':a.broadcast_modulation,'dtype':a.dtype,'seq':a.seq,'heads':a.heads,'seed':a.seed,'repeats':a.repeats,'samples':a.samples},
            'results':results,'diagnostic_only':True},indent=2)+'\n')
sys.exit(int(any(not r['correct'] for r in results)))
