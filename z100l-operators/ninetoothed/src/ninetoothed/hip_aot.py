"""HIP AOT for NineToothed-generated Triton, with a portable C tensor ABI.

Only the NineToothed generator supplies device code. This module emits host
module-loading code and never substitutes an external compute kernel.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ninetoothed import naming
from ninetoothed.generation import CodeGenerator
from ninetoothed.platform import backend_options_for_target, current_target
from ninetoothed.tensor import Tensor


_CTYPES = {"i1": "int32_t", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t",
           "i64": "int64_t", "u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t",
           "u64": "uint64_t", "fp32": "float", "fp64": "double"}


def _cexpr(node, constants):
    if isinstance(node, ast.Name):
        return str(constants[node.id]) if node.id in constants else node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return repr(node.value)
    if isinstance(node, ast.BinOp):
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
              ast.FloorDiv: "/", ast.Mod: "%"}.get(type(node.op))
        if op:
            return f"({_cexpr(node.left, constants)} {op} {_cexpr(node.right, constants)})"
    raise ValueError(f"Unsupported HIP AOT grid expression: {ast.dump(node)}")


def aot_hip(func, kernel_name, output_dir, num_warps=None, num_stages=None,
            *, backend_options=None, archs=None, _variant=None):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from ninetoothed.aot import _HEADER_CONTENT, _load_launch_func

    kernel_name = kernel_name or func.__name__
    if not re.fullmatch(r"[a-zA-Z_]\w*", kernel_name):
        raise ValueError("kernel_name must be a C identifier")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_warps = 4 if num_warps is None else num_warps
    num_stages = 1 if num_stages is None else num_stages
    target = current_target()
    if target.backend != "hip":
        raise RuntimeError("HIP AOT requires a HIP device for NineToothed shape generation")
    archs = tuple(dict.fromkeys(archs or [target.arch]))
    for arch in archs:
        if not re.fullmatch(r"gfx[0-9a-f]+", arch):
            raise ValueError(f"Invalid HIP architecture: {arch!r}")

    if _variant is None:
        # Specialize only after a runtime stride check; never assume contiguous
        # pointers for sliced/transposed tensors. Generic variant remains valid.
        for suffix in ("generic", "inner1", "aligned"):
            aot_hip(func, kernel_name + "_" + suffix, output_dir, num_warps, num_stages,
                    backend_options=backend_options, archs=archs, _variant=suffix)
        sources = [t.source for t in func.__annotations__.values()]
        names = [naming.remove_prefixes(t.name) for t in sources]
        params = ", ".join(["NineToothedStream stream"] +
                           [f"NineToothedTensor {n}" for n in names])
        export = f"NineToothedResult launch_{kernel_name}({params})"
        checks = [f"{n}.strides[{t.ndim-1}] == 1" for n, t in zip(names, sources) if t.ndim]
        aligned_checks = list(checks)
        for name, tensor in zip(names, sources):
            if tensor.ndim:
                aligned_checks.append(f"(reinterpret_cast<uintptr_t>({name}.data) % 16) == 0")
                aligned_checks.extend(f"({name}.strides[{d}] % 16) == 0" for d in range(tensor.ndim-1))
                if tensor.ndim > 1:
                    aligned_checks.append(f"{name}.shape[0] == 1")
        call_args = ", ".join(["stream"] + names)
        (output_dir / f"{kernel_name}.h").write_text(
            f'#pragma once\n#include "ninetoothed.h"\n#ifdef __cplusplus\nextern "C"\n#endif\n{export};\n')
        dispatch = (f'#include "{kernel_name}.h"\n#include "{kernel_name}_generic.h"\n'
                    f'#include "{kernel_name}_inner1.h"\n#include "{kernel_name}_aligned.h"\nextern "C" {export} {{\n'
                    f'    if ({" && ".join(aligned_checks) or "true"}) return launch_{kernel_name}_aligned({call_args});\n'
                    f'    if ({" && ".join(checks) or "true"}) return launch_{kernel_name}_inner1({call_args});\n'
                    f'    return launch_{kernel_name}_generic({call_args});\n}}\n')
        dispatch_path = output_dir / f"{kernel_name}.hip.cpp"
        dispatch_path.write_text(dispatch)
        hipcc = os.environ.get("HIPCC") or shutil.which("hipcc")
        if not hipcc:
            raise RuntimeError("hipcc not found; source /opt/dtk/env.sh or set HIPCC")
        subprocess.run([hipcc, "--offload-arch=" + target.arch, "-std=c++17", "-O2",
                        "-shared", "-fPIC", "-pthread", str(dispatch_path),
                        str(output_dir / f"{kernel_name}_generic.hip.cpp"),
                        str(output_dir / f"{kernel_name}_inner1.hip.cpp"),
                        str(output_dir / f"{kernel_name}_aligned.hip.cpp"),
                        "-o", str(output_dir / f"{kernel_name}.so")], check=True)
        return _load_launch_func(kernel_name, output_dir)

    generator = CodeGenerator()
    source_path = generator(func, "cuda", kernel_name, num_warps, num_stages,
                            None, False, backend_options=backend_options)
    source_bytes = Path(source_path).read_bytes()
    (output_dir / f"{kernel_name}.generated.py").write_bytes(source_bytes)
    module_name = "_nt_hip_" + hashlib.sha256(source_bytes).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)
    tensors = {naming.remove_prefixes(t.source.name): t.source for t in generator.tensors}
    signature, constants, declarations, arg_names, aligned_indices = {}, {}, [], [], set()
    for index, arg in enumerate(generator.kernel_func.args.args):
        name = arg.arg
        ptr = Tensor.pointer_pattern().fullmatch(name)
        size = Tensor.size_pattern().fullmatch(name)
        stride = Tensor.stride_pattern().fullmatch(name)
        if ptr:
            source = naming.remove_prefixes(ptr[1])
            dtype = "*" + str(tensors[source].dtype)
            expr = source + ".data"
            if _variant == "aligned":
                aligned_indices.add(index)
        elif size or stride:
            match = size or stride
            source = naming.remove_prefixes(match[1])
            if stride and _variant in ("inner1", "aligned") and int(match[3]) == tensors[source].ndim - 1:
                constants[name] = 1
                continue
            if size and _variant == "aligned" and tensors[source].ndim > 1 and int(match[3]) == 0:
                constants[name] = 1
                continue
            if stride and _variant == "aligned":
                aligned_indices.add(index)
            dtype = "i64"
            expr = f"{source}.{'shape' if size else 'strides'}[{match[3]}]"
            if arg.annotation:
                raise ValueError("HIP AOT needs explicit fixed shape values, not symbolic constexpr dimensions")
        else:
            source = naming.remove_prefixes(name)
            if source not in tensors:
                raise ValueError(f"Unresolved AOT parameter {name}; use fixed block sizes and shapes")
            tensor = tensors[source]
            if tensor.constexpr:
                constants[name] = tensor.value
                continue
            dtype = str(tensor.dtype)
            expr = f"*static_cast<{_CTYPES[dtype]} *>({source}.data)"
        signature[name] = dtype
        ctype = "void *" if dtype.startswith("*") else _CTYPES[dtype]
        declarations.append(f"    {ctype} arg_{name} = ({ctype})({expr});")
        arg_names.append(name)

    grid = _cexpr(copy.deepcopy(generator.raw_grid.body.elts[0]), constants)
    for name in arg_names:
        grid = re.sub(r"\b" + re.escape(name) + r"\b", "arg_" + name, grid)
    launch_names = [a.arg for a in generator.launch_func.args.args]
    params = ", ".join(["NineToothedStream stream"] +
                       [f"NineToothedTensor {n}" for n in launch_names])
    export = f"NineToothedResult launch_{kernel_name}({params})"
    blobs, choices, metadata = [], [], []
    for arch in archs:
        gpu = GPUTarget("hip", arch, target.warp_size)
        options = {"num_warps": num_warps, "num_stages": num_stages,
                   **backend_options_for_target(backend_options, gpu)}
        attrs = None
        source_args = {"fn": kernel, "signature": signature}
        if "constexprs" in inspect.signature(ASTSource).parameters:
            # Triton 3.8 supports gfx906 and uses tuple-indexed attributes.
            from triton.compiler.compiler import make_backend
            attrs = {(i,): make_backend(gpu).parse_attr("D") for i in aligned_indices}
            source_args["constexprs"] = constants
        else:
            if aligned_indices:
                from triton.compiler.compiler import AttrsDescriptor
                attrs = AttrsDescriptor(divisible_by_16=aligned_indices)
            source_args["constants"] = constants
        compiled = triton.compile(ASTSource(**source_args, attrs=attrs),
                                  target=gpu, options=options)
        hsaco = compiled.asm["hsaco"]
        hsaco_path = output_dir / f"{kernel_name}.{arch}.hsaco"
        hsaco_path.write_bytes(hsaco)
        (output_dir / f"{kernel_name}.{arch}.amdgcn").write_text(compiled.asm["amdgcn"])
        byte_lines = [", ".join(str(x) for x in hsaco[i:i+32]) for i in range(0, len(hsaco), 32)]
        blobs.append(f"static const unsigned char image_{arch}[] = {{\n" + ",\n".join(byte_lines) + "\n};")
        m = compiled.metadata
        if hasattr(m, "global_scratch_size"):
            if m.global_scratch_size or m.profile_scratch_size:
                raise NotImplementedError("HIP AOT requires zero global/profile scratch")
        choices.append(f'''        if (arch == "{arch}") {{
            image = image_{arch}; symbol = "{m.name}";
            entry.shared = {m.shared}; entry.threads = {m.num_warps * gpu.warp_size};
        }}''')
        metadata.append({"arch": arch, "kernel_symbol": m.name, "shared_bytes": m.shared,
                         "threads": m.num_warps * gpu.warp_size, "options": options,
                         "hsaco_sha256": hashlib.sha256(hsaco).hexdigest()})
    # Local includes make artifacts relocatable to another checkout or machine.
    (output_dir / "ninetoothed.h").write_text(_HEADER_CONTENT)
    (output_dir / f"{kernel_name}.h").write_text(
        f'#pragma once\n#include "ninetoothed.h"\n#ifdef __cplusplus\nextern "C"\n#endif\n{export};\n')
    cpp = '''#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <limits>
#include "KERNEL.h"

namespace {
BLOBS
struct Entry { hipModule_t module = nullptr; hipFunction_t function = nullptr;
               unsigned shared = 0, threads = 0; };
std::mutex cache_mutex;
std::map<int, Entry> entries;
hipError_t get_entry(Entry &result) {
    int device = -1;
    hipError_t err = hipGetDevice(&device);
    if (err != hipSuccess) return err;
    std::lock_guard<std::mutex> guard(cache_mutex);
    auto found = entries.find(device);
    if (found != entries.end()) { result = found->second; return hipSuccess; }
    Entry entry;
    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device);
    if (err != hipSuccess) return err;
    std::string arch(props.gcnArchName);
    arch = arch.substr(0, arch.find(':'));
    const void *image = nullptr;
    const char *symbol = nullptr;
CHOICES
    if (!image) return hipErrorNoBinaryForGpu;
    err = hipModuleLoadData(&entry.module, image);
    if (err != hipSuccess) return err;
    err = hipModuleGetFunction(&entry.function, entry.module, symbol);
    if (err != hipSuccess) { (void)hipModuleUnload(entry.module); return err; }
    entries.emplace(device, entry);
    result = entry;
    return hipSuccess;
}
} // namespace

extern "C" EXPORT {
DECLARATIONS
    const int64_t grid = GRID;
    if (grid == 0) return hipSuccess;
    if (grid < 0 || static_cast<uint64_t>(grid) > std::numeric_limits<unsigned>::max())
        return hipErrorInvalidValue;
    Entry entry;
    hipError_t err = get_entry(entry);
    if (err != hipSuccess) return err;
    void *args[] = {ARG_POINTERS};
    return hipModuleLaunchKernel(entry.function, static_cast<unsigned>(grid), 1, 1,
        entry.threads, 1, 1, entry.shared, static_cast<hipStream_t>(stream), args, nullptr);
}
'''
    argument_pointers = ["&arg_" + n for n in arg_names]
    if hasattr(compiled.metadata, "global_scratch_size"):
        # Modern Triton adds these two ABI parameters even at zero size.
        # Omitting them makes HIP read past the host parameter array.
        declarations += ["    void *arg_global_scratch = nullptr;",
                         "    void *arg_profile_scratch = nullptr;"]
        argument_pointers += ["&arg_global_scratch", "&arg_profile_scratch"]
    for key, value in {"KERNEL": kernel_name, "BLOBS": "\n".join(blobs),
                       "CHOICES": "\n".join(choices), "EXPORT": export,
                       "DECLARATIONS": "\n".join(declarations), "GRID": grid,
                       "ARG_POINTERS": ", ".join(argument_pointers)}.items():
        cpp = cpp.replace(key, value)
    cpp_path = output_dir / f"{kernel_name}.hip.cpp"
    cpp_path.write_text(cpp)
    manifest = {"generator": "ninetoothed", "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "signature": signature, "constants": constants, "targets": metadata}
    (output_dir / f"{kernel_name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
