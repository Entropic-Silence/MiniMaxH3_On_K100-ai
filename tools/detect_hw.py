#!/usr/bin/env python3
"""Read-only hardware inventory. Does not import or upgrade model dependencies."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def detect():
    cgroup_paths = ["memory.max", "memory.current", "cpu.max", "cpuset.cpus.effective",
                    "memory/memory.limit_in_bytes", "cpu/cpu.cfs_quota_us",
                    "cpu/cpu.cfs_period_us", "cpuset/cpuset.cpus"]
    cg = {p: read("/sys/fs/cgroup/" + p) for p in cgroup_paths}
    limits = [int(cg[p]) for p in ("memory.max", "memory/memory.limit_in_bytes")
              if cg[p] and cg[p].isdigit() and int(cg[p]) < 2**60]
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    quota = None
    if cg["cpu.max"]:
        q, period = cg["cpu.max"].split()
        if q != "max":
            quota = int(q) / int(period)
    elif cg["cpu/cpu.cfs_quota_us"] and int(cg["cpu/cpu.cfs_quota_us"]) > 0:
        quota = int(cg["cpu/cpu.cfs_quota_us"]) / int(cg["cpu/cpu.cfs_period_us"])
    packages = {}
    for name in ("torch", "triton", "transformers", "diffusers", "safetensors", "numpy",
                 "ninetoothed", "accelerate", "av", "scikit-image", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hostname": platform.node(), "python": platform.python_version(),
              "kernel": platform.release(), "packages": packages, "cgroup": cg,
              "host_memory_limit_bytes": min(limits) if limits else None,
              "effective_cpu_count": min(cpus, quota) if quota else cpus,
              "dtk_path": str(Path("/opt/dtk").resolve()), "devices": []}
    try:
        import torch
        result["hip_version"] = torch.version.hip
        result["cxx11_abi"] = torch._C._GLIBCXX_USE_CXX11_ABI
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            result["devices"].append({"index": i, "name": p.name,
                "arch": getattr(p, "gcnArchName", "").split(":")[0],
                "gcn_arch_name": getattr(p, "gcnArchName", None),
                "total_memory_bytes": total, "free_memory_bytes": free,
                "compute_units": p.multi_processor_count})
        result["p2p_access"] = [[i == j or torch.cuda.can_device_access_peer(i, j)
            for j in range(torch.cuda.device_count())] for i in range(torch.cuda.device_count())]
    except Exception as exc:
        result["torch_error"] = repr(exc)
    smi = shutil.which("hy-smi") or shutil.which("rocm-smi")
    if smi:
        result["smi"] = command([smi, "--showproductname", "--showmeminfo", "vram", "--showtopo"])
    result["xmake"] = command(["xmake", "--version"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(detect(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
