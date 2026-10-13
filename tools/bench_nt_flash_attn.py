import time, torch, sys, traceback, functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(q, k, v, scale, o, block_size_m=None, block_size_n=None):
    def arr_qo(x):
        a = x.tile((1, 1, block_size_m, -1))
        a.dtype = a.dtype.squeeze((0, 1))
        return a
    def arr_kv(x):
        a = x.tile((1, 1, block_size_n, -1)).tile((1, 1, -1, -1)).expand((-1, -1, q_arr.shape[-2], -1))
        a.dtype = a.dtype.squeeze((0, 1, 3))
        a.dtype.dtype = a.dtype.dtype.squeeze((0, 1))
        return a
    q_arr = arr_qo(q)
    return q_arr, arr_kv(k), arr_kv(v), scale, arr_qo(o)

def application(q, k, v, scale, o):
    q_i = (1.4426950408889634 * scale * q).to(q.dtype)
    acc = ntl.zeros((q_i.shape[-2], q_i.shape[-1]), dtype=ntl.float32)
    l = ntl.full((q_i.shape[-2],), 1, dtype=ntl.float32)
    m = ntl.full((q_i.shape[-2],), float("-inf"), dtype=ntl.float32)
    for j in range(k.shape[0]):
        qk = ntl.dot(q_i, ntl.trans(k[j]))
        qk = ntl.where(k[j].offsets(-2) < k.source.shape[-2], qk, float("-inf"))
        m_new = ntl.maximum(m, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_new[:, None])
        alpha = ntl.exp2(m - m_new)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v[j].dtype), v[j])
        m = m_new
        l = l * alpha + ntl.sum(p, 1)
    acc /= l[:, None]
    o = acc  # noqa

def bench(fn, it=3):
    fn(); torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/it

S=int(sys.argv[1]) if len(sys.argv)>1 else 16384; H=14; D=128
q=torch.randn(1,H,S,D,device='cuda',dtype=torch.bfloat16); k=torch.randn_like(q); v=torch.randn_like(q)
ref=torch.nn.functional.scaled_dot_product_attention(q,k,v)
import os
CFGS=[tuple(map(int,c.split(","))) for c in os.environ.get("CFGS","128,32,4,1").split(";")]
for bm,bn,nw,ns in CFGS:
    try:
        shp=(None,None,None,{"constexpr": True, "upper_bound": 128})
        tensors=(Tensor(4,dtype=ninetoothed.bfloat16,shape_options=shp),Tensor(4,dtype=ninetoothed.bfloat16,shape_options=shp),Tensor(4,dtype=ninetoothed.bfloat16,shape_options=shp),Tensor(0,dtype=ninetoothed.float64),Tensor(4,dtype=ninetoothed.bfloat16,shape_options=shp))
        kern = ninetoothed.make(functools.partial(arrangement, block_size_m=bm, block_size_n=bn), application, tensors, num_warps=nw, num_stages=ns)
        out=torch.empty_like(q)
        fn=lambda: kern(q,k,v,1.0/D**0.5,out)
        t=bench(fn)
        err=(out.float()-ref.float()).abs().max().item()
        print(f"nt fa bm={bm} bn={bn} nw={nw} ns={ns}: {t*1e3:.1f} ms {4*S*S*D*H/t/1e12:.1f} TFLOPS maxerr={err:.3e}", flush=True)
    except Exception as e:
        print("fail", bm,bn,nw,ns, repr(e)[:400])
t=bench(lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v))
print(f"torch sdpa: {t*1e3:.1f} ms {4*S*S*D*H/t/1e12:.1f} TFLOPS")
