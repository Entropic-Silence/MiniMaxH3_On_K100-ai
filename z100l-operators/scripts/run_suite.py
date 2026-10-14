#!/usr/bin/env python3
"""Operator-only qualification on the contest DCU; never loads a model."""
import argparse
import gc
import json
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
a.output.mkdir(parents=True,exist_ok=True)
benchmark=root/'tools/bench_gfx906_pointwise.py'
cases=[]
for seq in [17,128,8192]:
 for seed in [0,1]:
  measured=seq==8192 and seed==0
  cases.append((f'ops_bf16_s{seq}_seed{seed}',seq,seed,30 if measured else 3,5 if measured else 1,None,False))
for seq,seed in [(17,1),(8192,0)]:
 cases.append((f'broadcast_bf16_s{seq}_seed{seed}',seq,seed,30 if seq==8192 else 3,5 if seq==8192 else 1,
              'rms_modulate,gated_residual_rms_modulate',True))
summary=[]
for tag,seq,seed,repeats,samples,ops,broadcast in cases:
 argv=[str(benchmark),'--seq',str(seq),'--seed',str(seed),'--dtype','bf16',
       '--repeats',str(repeats),'--samples',str(samples),'--caller','hip',
       '--aot-dir',str(a.output/'aot'),'--json',str(a.output/(tag+'.json'))]
 if ops:argv+=['--ops',ops]
 if broadcast:argv+=['--broadcast-modulation']
 print('CASE',tag,flush=True)
 sys.argv=argv
 try:
  runpy.run_path(str(benchmark),run_name='__main__')
  status=0
 except SystemExit as exc:
  status=int(exc.code or 0)
 result=json.loads((a.output/(tag+'.json')).read_text())
 summary.append({'case':tag,'exit_code':status,'operators':len(result['results']),
                 'correct':sum(r.get('correct',False) for r in result['results']),
                 'bitwise':sum(r.get('bitwise_equal',False) for r in result['results'])})
 (a.output/'suite_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 gc.collect()
 import torch
 torch.cuda.empty_cache()
print('SUITE_SUMMARY',json.dumps(summary),flush=True)
raise SystemExit(int(any(row['exit_code'] for row in summary)))
