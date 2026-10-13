#!/usr/bin/env python3
"""Diagnostic: kernel sequence of one DiT block (device 0) from a hipprof db."""
import sqlite3
import sys


def main():
    db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    c = sqlite3.connect(db)
    table = [n for (n,) in c.execute("select name from sqlite_master where name like 'OPS_%'")][0]
    strings = {str(i): n for i, n in c.execute("select STR_ID, STR_NAME from STR_TABLE")}
    ops = sorted((b, e, q, strings.get(str(n), str(n))) for b, e, q, n in
                 c.execute(f"select BeginNs, EndNs, queue_id, Name from {table} where dev_id=?", (dev,)))
    fa = [o for o in ops if "flash_fwd" in o[3] or o[3].startswith("h3_full_attention")]
    # Block boundary: the gated residual + norm kernel that starts each MLP half; pick one mid-forward.
    anchor = fa[len(fa) * 3 // 4][0]
    starts = [o[0] for o in ops if "gated_residual_exact" in o[3] and o[0] < anchor]
    begin, end = starts[-2], starts[-1]
    t0 = begin
    for b, e, q, n in ops:
        if begin <= b < end:
            short = n.split("(")[0].replace("void ", "")[:48]
            print(f"q{q:>3} {(b - t0) / 1e6:8.2f} -> {(e - t0) / 1e6:8.2f} ms ({(e - b) / 1e6:6.2f})  {short}")


if __name__ == "__main__":
    main()
