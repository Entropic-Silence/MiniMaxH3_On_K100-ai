import sqlite3, sys
db, dev, block = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 25
c = sqlite3.connect(db)
table = [n for (n,) in c.execute("select name from sqlite_master where name like 'OPS_%'")][0]
strings = {str(i): n for i, n in c.execute("select STR_ID, STR_NAME from STR_TABLE")}
ops = sorted((b, e, q, strings.get(str(n), str(n))) for b, e, q, n in
             c.execute(f"select BeginNs, EndNs, queue_id, Name from {table} where dev_id=?", (dev,)))
fa = [o for o in ops if "flash_fwd" in o[3]]
# last forward: 100 FA per forward (2 groups x 50 blocks)
fa = fa[-100:]
t_start = fa[2*block][0] - 30_000_000
t_end = fa[2*block+2][0] if 2*block+2 < len(fa) else fa[-1][1]
sel = [o for o in ops if t_start <= o[0] < t_end]
t0 = sel[0][0]
for b, e, q, n in sel:
    print(f"{(b-t0)/1e6:8.2f} {(e-t0)/1e6:8.2f} {(e-b)/1e6:7.2f}ms q{q} {n[:70]}")
