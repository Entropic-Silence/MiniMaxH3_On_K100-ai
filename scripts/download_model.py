#!/usr/bin/env python3
"""Preparation only: download the pinned official modular FL2VA checkpoint."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import time
from huggingface_hub import HfApi, snapshot_download

REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PATTERNS = ["model_index.json", "modular_model_index.json", "processor/*", "tokenizer/*",
            "text_encoder/*", "transformer/*", "vae/*", "audio_vae/*",
            "scheduler/*", "audio_scheduler/*"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "download_started", "repo": "MiniMaxAI/MiniMax-H3",
                      "revision": REVISION, "directory": str(args.directory)}), flush=True)
    for attempt in ([] if args.verify_only else range(1, 6)):
        try:
            snapshot_download("MiniMaxAI/MiniMax-H3", revision=REVISION,
                              allow_patterns=PATTERNS, local_dir=args.directory,
                              max_workers=args.workers)
            break
        except Exception as exc:
            # Hub exceptions can contain signed URLs; retain the type, not tokens.
            print(json.dumps({"event": "download_failed", "attempt": attempt,
                              "error_type": type(exc).__name__}), flush=True)
            if attempt == 5:
                raise SystemExit(1) from None
            time.sleep(10)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if manifest["revision"] != REVISION:
            raise ValueError("Manifest revision differs from pinned checkpoint")
        expected = {f["rfilename"]: f for f in manifest["files"]}
    else:
        info = HfApi().model_info("MiniMaxAI/MiniMax-H3", revision=REVISION, files_metadata=True)
        expected = {f.rfilename: {"size": f.size, "blobId": f.blob_id,
                                  "lfs": {"sha256": f.lfs.sha256} if f.lfs else None}
                    for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in PATTERNS)}
    missing = [name for name in expected if not (args.directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete checkpoint: {len(missing)} missing files: {missing}")
    records = []
    for p in sorted(args.directory.rglob("*")):
        if not p.is_file() or ".cache" in p.parts or p.name == "SHA256SUMS.json":
            continue
        name = p.relative_to(args.directory).as_posix()
        if name not in expected:
            continue
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {p.stat().st_size}\0".encode())
        with p.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(block)
                if not expected[name].get("lfs"):
                    blob.update(block)
        actual = digest.hexdigest()
        record = expected[name]
        if p.stat().st_size != record["size"] or (record.get("lfs") and actual != record["lfs"]["sha256"]):
            raise RuntimeError(f"Checkpoint verification failed: {name}")
        if not record.get("lfs") and record.get("blobId") and blob.hexdigest() != record["blobId"]:
            raise RuntimeError(f"Checkpoint Git blob verification failed: {name}")
        records.append({"path": name, "bytes": p.stat().st_size, "sha256": actual})
        print(json.dumps({"event": "verified", **records[-1]}), flush=True)
    if len(records) != len(expected):
        raise RuntimeError("Not all manifest entries were verified")
    (args.directory / "SHA256SUMS.json").write_text(json.dumps(
        {"revision": REVISION, "files": records}, indent=2) + "\n")
    print(json.dumps({"event": "complete", "files": len(records)}), flush=True)


if __name__ == "__main__":
    main()
