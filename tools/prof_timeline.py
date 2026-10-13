#!/usr/bin/env python3
"""Diagnostic: per-queue busy time, overlap and idle gaps in the last DiT
forward of a hipprof database (device 0)."""
import sqlite3
import sys
from collections import defaultdict


def union(intervals):
    total, end = 0, None
    merged = []
    for b, e in sorted(intervals):
        if merged and b <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([b, e])
    return merged, sum(e - b for b, e in merged)


def main():
    db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    c = sqlite3.connect(db)
    table = [n for (n,) in c.execute("select name from sqlite_master where name like 'OPS_%'")][0]
    strings = {str(i): n for i, n in c.execute("select STR_ID, STR_NAME from STR_TABLE")}
    ops = [(b, e, q, strings.get(str(n), str(n))) for b, e, q, n in
           c.execute(f"select BeginNs, EndNs, queue_id, Name from {table} where dev_id=?", (dev,))]
    fa = sorted(b for b, e, q, n in ops
                if n.startswith("h3_full_attention") or "flash_fwd" in n or "fmha" in n)
    # The last forward: FA launches cluster per forward; split at the largest gap.
    gaps = [(fa[i + 1] - fa[i], i) for i in range(len(fa) - 1)]
    _, cut = max(gaps)
    start = fa[cut + 1] - 200_000_000  # include the pre-attention work of block 0
    ops = [o for o in ops if o[0] >= start]
    end = max(e for b, e, q, n in ops)
    print(f"window {(end - start) / 1e9:.3f} s, ops {len(ops)}")
    by_queue = defaultdict(list)
    names = defaultdict(lambda: defaultdict(int))
    for b, e, q, n in ops:
        by_queue[q].append((b, e))
        names[q][n.split("(")[0][:50]] += e - b
    for q, iv in by_queue.items():
        _, busy = union(iv)
        top = sorted(names[q].items(), key=lambda kv: -kv[1])[:4]
        print(f"queue {q}: busy {busy / 1e9:.3f} s ({len(iv)} ops) top {[(k, round(v / 1e9, 3)) for k, v in top]}")
    merged, busy = union([(b, e) for b, e, q, n in ops])
    print(f"any-queue busy {busy / 1e9:.3f} s, idle {(end - start - busy) / 1e9:.3f} s")
    comm = [(b, e) for b, e, q, n in ops if n.startswith("nccl")]
    compute = [(b, e) for b, e, q, n in ops if not n.startswith("nccl")]
    cm, cbusy = union(compute)
    mm, mbusy = union(comm)
    exposed = busy - cbusy
    print(f"compute busy {cbusy / 1e9:.3f} s, comm busy {mbusy / 1e9:.3f} s, comm-only (exposed) {exposed / 1e9:.3f} s")
    idle = sorted(((cm[i + 1][0] - cm[i][1]), cm[i][1]) for i in range(len(cm) - 1))[-5:]
    print("largest compute gaps (ms):", [round(g / 1e6, 2) for g, _ in idle])


if __name__ == "__main__":
    main()
