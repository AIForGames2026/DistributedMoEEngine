"""Checks a recording's per-call cache statuses and prefetch hints (TASK-41).

For every step after the first, the counts of each status in `c` must equal the change in the
engine's own `cache` counters. Prints the hit rates over the whole recording and per turn, and
how many hints were sent and how many of them brought an expert in from the drive.

Usage: python check_cache_frames.py RECORDING.jsonl[.gz] ...
"""
import gzip
import json
import sys

NAMES = ["", "resident", "hinted", "late", "missed"]


def frames(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def check(path):
    prev = None
    steps = matched = 0
    total = [0] * 5
    hints = loads = 0
    turns = []
    for f in frames(path):
        if f.get("t") == "chat":
            turns.append([0] * 5)
        if f.get("t") != "step":
            continue
        steps += 1
        k = [0] * 5
        status = {}
        for i in range(0, len(f.get("c", [])), 3):
            k[f["c"][i + 2]] += 1
            status[(f["c"][i], f["c"][i + 1])] = f["c"][i + 2]
        h = f.get("h", [])
        hints += len(h) // 3
        loads += sum(1 for i in range(0, len(h), 3) if status.get((h[i], h[i + 1])) in (2, 3))
        for i in range(5):
            total[i] += k[i]
            if turns:
                turns[-1][i] += k[i]
        c = f["cache"]
        if prev is not None:
            delta = [c[n] - prev[n] for n in NAMES[1:]]
            if delta != k[1:]:
                raise SystemExit(f"{path}: step {f['seq']} has statuses {k[1:]} but the counters moved by {delta}")
            matched += 1
        prev = c
    calls = sum(total)

    def rates(t):
        n = sum(t) or 1
        return ", ".join(f"{NAMES[i]} {100 * t[i] / n:.1f}%" for i in range(1, 5))

    print(f"{path}: {steps} steps, {matched} of {steps - 1} checked against the engine's counters (all match)")
    print(f"  {calls} calls: {rates(total)}")
    for i, t in enumerate(turns):
        print(f"  turn {i + 1}: {sum(t)} calls: {rates(t)}")
    print(f"  {hints} hints sent, {loads} of them brought an expert in from the drive")


for p in sys.argv[1:]:
    check(p)
