import time, torch, os
torch.backends.cuda.matmul.allow_tf32=False
dev='cuda'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
def bench(fn, it=5):
    fn(); torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/it
S=40000
for (M,N,K) in [(S,3*7168,5376),(S,5376,7168),(S,2*14336,5376),(S,5376,14336)]:
    a=torch.randn(M,K,device=dev,dtype=torch.bfloat16); b=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
    t=bench(lambda: a@b.t())
    print(f"gemm {M}x{N}x{K}: {t*1e3:.1f} ms  {2*M*N*K/t/1e12:.1f} TFLOPS")
    del a,b
x=torch.randn(S*5376*4//2,device=dev,dtype=torch.bfloat16)
t=bench(lambda: x.mul_(1.0001))
print(f"elementwise rw bw: {2*x.numel()*2/t/1e9:.0f} GB/s")
del x
import torch.nn.functional as F
for H in (14,):
    q=torch.randn(1,H,S,128,device=dev,dtype=torch.bfloat16); k=torch.randn_like(q); v=torch.randn_like(q)
    try:
        t=bench(lambda: F.scaled_dot_product_attention(q,k,v),2)
        print(f"torch sdpa H={H} S={S}: {t*1e3:.0f} ms {4*S*S*128*H/t/1e12:.1f} TFLOPS")
    except Exception as e: print("sdpa fail", repr(e)[:200])
    try:
        from flash_attn import flash_attn_func
        qq,kk,vv=[z.transpose(1,2).contiguous() for z in (q,k,v)]
        t=bench(lambda: flash_attn_func(qq,kk,vv),2)
        print(f"flash_attn H={H} S={S}: {t*1e3:.0f} ms {4*S*S*128*H/t/1e12:.1f} TFLOPS")
    except Exception as e: print("fa fail", repr(e)[:200])
