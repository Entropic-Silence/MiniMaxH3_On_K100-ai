"""Wrap the unchanged official CLI; retain stdout, timing and SMI samples."""
import datetime
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def official_compare():
    path = ROOT / "InfiniLM/examples/minimax_h3_compare.py"
    spec = importlib.util.spec_from_file_location("h3_official_compare", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    directory = Path(os.environ.get("H3_LOG_DIR", ROOT / "logs" /
        datetime.datetime.now().strftime("%Y-%m-%d/%H%M%S-case")))
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(ROOT / "InfiniLM/examples/minimax_h3.py"), *sys.argv[1:]]
    (directory / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2)+"\n")
    runtime = {"environment": {name: os.environ.get(name) for name in (
        "H3_TEXT_ENCODER", "H3_ADALN_NUMERICS", "H3_POINTWISE_NUMERICS", "H3_FUSED_ADALN",
        "H3_DETERMINISTIC_BLAS", "H3_ROPE_NUMERICS",
        "H3_QWEN_FP32_ROPE", "H3_QWEN_NINETOOTHED", "H3_QWEN_TRACE_DIR",
        "H3_TRACE_DIR", "H3_TRACE_TOKENS", "INFINICORE_HYGON_MHA",
        "INFINI_ROOT", "CUDA_VISIBLE_DEVICES")},
        "packages": {}}
    runtime["source_pins"] = json.loads((ROOT / "versions.json").read_text())
    runtime["native_binaries"] = {}
    for binary_dir in (ROOT / "InfiniCore/python/infinicore/lib", ROOT / "InfiniLM/python/infinilm/lib",
                      Path(os.environ.get("INFINI_ROOT", ROOT / ".infini")) / "lib"):
        for file in sorted(binary_dir.glob("*.so")):
            digest = hashlib.sha256()
            with file.open("rb") as source:
                for block in iter(lambda:source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            runtime["native_binaries"][str(file)] = digest.hexdigest()
    for name in ("torch", "triton", "numpy", "diffusers", "transformers", "huggingface-hub", "safetensors"):
        try:
            runtime["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime["packages"][name] = None
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2)+"\n")
    stop = threading.Event()
    smi = shutil.which("hy-smi") or shutil.which("rocm-smi")
    def sample():
        if not smi:
            return
        with (directory / "smi.jsonl").open("w") as log:
            while not stop.is_set():
                try:
                    r = subprocess.run([smi,"--showmeminfo","vram"], capture_output=True,text=True,timeout=5)
                    log.write(json.dumps({"time":time.time(),"returncode":r.returncode,"stdout":r.stdout})+"\n")
                    log.flush()
                except (OSError,subprocess.TimeoutExpired) as exc:
                    log.write(json.dumps({"error":repr(exc)})+"\n")
                stop.wait(1)
    compare = official_compare()
    worker = threading.Thread(target=sample,daemon=True)
    timings, returncode, nfe = {}, None, None
    started = time.perf_counter()
    worker.start()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        with (directory / "stdout.log").open("w") as log:
            child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,bufsize=1,env=env)
            try:
                for line in child.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                    compare.parse_timing_line(line,timings)
                    if "compute.h3_transformer" in line:
                        match = re.search(r"\((\d+) calls,", line)
                        if match:
                            nfe = int(match.group(1))
                returncode = child.wait()
            except BaseException:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
    finally:
        stop.set()
        worker.join(timeout=6)
        (directory / "timing.json").write_text(json.dumps({"returncode":returncode,
            "process_wall_seconds":time.perf_counter()-started,"timings":timings,
            "observed_transformer_calls":nfe},indent=2)+"\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
