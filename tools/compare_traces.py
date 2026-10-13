#!/usr/bin/env python3
"""Diagnostic: compare two tools/trace_pipeline.py traces call by call and
report the first call (in execution order) whose inputs or output differ.
Never used by generation.

usage: compare_traces.py A.jsonl B.jsonl [C.jsonl ...]   (each against A)
"""
import json
import sys


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    print(f"{b_path} vs {a_path}: {len(b)} / {len(a)} calls")
    for ra, rb in zip(a, b):
        name = f"{ra['call']}#{ra['index']}"
        if (ra["call"], ra["index"]) != (rb["call"], rb["index"]):
            print(f"  call order differs at {name} vs {rb['call']}#{rb['index']}")
            return
        for part in ("args", "kwargs", "output"):
            fa, fb = dict(flatten(ra[part])), dict(flatten(rb[part]))
            differing = [key for key in fa if fa.get(key) != fb.get(key)]
            if differing:
                print(f"  first difference: {name} {part}: {', '.join(differing[:6])}")
                return
    print("  identical")


for other in sys.argv[2:]:
    compare(sys.argv[1], other)
