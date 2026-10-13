import triton, ctypes, torch
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import importlib.util, sys
spec = importlib.util.spec_from_file_location("aot_add", "" + __import__("os").path.join(__import__("os").path.dirname(__file__), "aot_probe_kernel.py") + ""); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
src = ASTSource(fn=m.add_kernel, signature={"x_ptr":"*fp32","y_ptr":"*fp32","o_ptr":"*fp32","n":"i32"}, constants={"BLOCK":1024})
ck = triton.compile(src, target=GPUTarget("hip", arch, 64), options={"num_warps":4})
print("asm keys", list(ck.asm.keys()), "name", ck.metadata.name, "shared", ck.metadata.shared, "warps", ck.metadata.num_warps)
hsaco = ck.asm["hsaco"]; open("add.hsaco","wb").write(hsaco)
# load with HIP module API through ctypes
hip = ctypes.CDLL("libamdhip64.so")
mod = ctypes.c_void_p(); fn = ctypes.c_void_p()
buf = ctypes.create_string_buffer(hsaco, len(hsaco))
print("hipModuleLoadData", hip.hipModuleLoadData(ctypes.byref(mod), buf))
print("hipModuleGetFunction", hip.hipModuleGetFunction(ctypes.byref(fn), mod, ck.metadata.name.encode()))
n=1<<20; x=torch.randn(n,device="cuda"); y=torch.randn(n,device="cuda"); o=torch.empty_like(x)
args=[ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),ctypes.c_void_p(o.data_ptr()),ctypes.c_int32(n)]
arr=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(a),ctypes.c_void_p) for a in args])
r=hip.hipModuleLaunchKernel(fn,(n+1023)//1024,1,1,64*4,1,1,ck.metadata.shared,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),arr,None)
torch.cuda.synchronize(); print("launch", r, "maxerr", (o-(x+y)).abs().max().item())
