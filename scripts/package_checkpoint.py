#!/usr/bin/env python3
"""Create a development checkpoint, not a claim of contest acceptance."""
import hashlib
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts"
DEST.mkdir(exist_ok=True)
excluded = {".git", ".xmake", ".pytest_cache", "__pycache__", "build", ".venv", ".infini", ".local"}


def source_filter(info):
    p = Path(info.name)
    if excluded.intersection(p.parts) or re.fullmatch(r"core\.\d+", p.name):
        return None
    if p.suffix in {".o", ".so", ".hsaco", ".pyc", ".mp4", ".pt", ".npy", ".bin"}:
        return None
    return info


source = DEST / "h3-contest-gfx936-source.tar.gz"
with tarfile.open(source, "w:gz") as archive:
    for name in [".gitignore", "README.md", "PROGRESS.md", "EXPERIMENTS.md", "COMPLIANCE.md",
                 "versions.json", "configs", "docs", "patches", "scripts", "tests", "tools", "logs", "samples",
                 "InfiniCore", "InfiniLM", "ninetoothed"]:
        archive.add(ROOT / name, arcname=name, filter=source_filter)
deps = DEST / "h3-contest-offline-deps.tar.gz"
with tarfile.open(deps, "w:gz") as archive:
    archive.add(ROOT / "third_party/wheels", arcname="third_party/wheels")
    archive.add(ROOT / "third_party/xmake-v3.1.1.gz.run", arcname="third_party/xmake-v3.1.1.gz.run")
records = []
for file in (source, deps):
    h = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    digest = h.hexdigest()
    records.append(f"{digest}  {file.name}")
    print(file.name, file.stat().st_size, digest)
(DEST / "SHA256SUMS").write_text("\n".join(records) + "\n")
