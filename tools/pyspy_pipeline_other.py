#!/usr/bin/env python3
"""Diagnostic: from a `py-spy record --format raw` profile of the unified
entry, attribute main-thread samples inside the generation call that are NOT
inside a timed component (the entry's ComponentProfiler `timed` wrapper) to
the pipeline block and leaf function they were spent in."""
import collections
import re
import sys


def main():
    path = sys.argv[1]
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    blocks = collections.Counter()
    leaves = collections.Counter()
    timed = collections.Counter()
    total = 0
    warmup = 0
    with open(path) as handle:
        for line in handle:
            line = line.rstrip()
            if not line:
                continue
            stack, _, count = line.rpartition(" ")
            count = int(count)
            frames = stack.split(";")
            if not any(f.startswith("main (") and "minimax_h3.py" in f for f in frames):
                continue
            if not any("modular_pipeline.py" in f and f.startswith("__call__") for f in frames):
                continue
            if any(f.startswith("warmup_pipeline (") for f in frames):
                warmup += count  # the load-time warmup generation, not timed
                continue
            total += count
            timed_index = next((i for i, f in enumerate(frames) if f.startswith("timed (")), None)
            if timed_index is not None:
                below = frames[timed_index + 1] if timed_index + 1 < len(frames) else "?"
                timed[re.sub(r":\d+\)", ")", below)] += count
                continue
            block = "?"
            for f in frames:
                if "modular_pipelines/minimax_h3/" in f and f.startswith("__call__"):
                    block = re.sub(r":\d+\)", ")", f)
            blocks[block] += count
            leaves[frames[-1]] += count
    other = total - sum(timed.values())
    print(f"generation samples {total} ({total / rate:.1f} s), timed {sum(timed.values()) / rate:.1f} s, "
          f"other {other / rate:.1f} s (warmup generation excluded: {warmup / rate:.1f} s)")
    print("timed components:")
    for name, count in timed.most_common(8):
        print(f"  {count / rate:7.2f} s  {name}")
    print("other, by pipeline block:")
    for name, count in blocks.most_common(15):
        print(f"  {count / rate:7.2f} s  {name}")
    print("other, by leaf frame:")
    for name, count in leaves.most_common(25):
        print(f"  {count / rate:7.2f} s  {name}")


if __name__ == "__main__":
    main()
