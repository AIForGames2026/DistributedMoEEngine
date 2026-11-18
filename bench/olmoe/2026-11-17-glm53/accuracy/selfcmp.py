#!/usr/bin/env python3
"""Shares of expert choices and DSA pools two engine runs have in common (TF_GOLDEN_DUMP files)."""
import sys
from array import array

LAYERS, K, DENSE, TOP = 45, 8, 3, 512
TOKENS = {0: 5, 1: 17, 2: 15, 3: 2600}


def load(path, code):
    a = array(code)
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def common(a, b, width, skip_unused):
    same = total = 0
    for r in range(len(a) // width):
        x = a[r * width:(r + 1) * width]
        y = b[r * width:(r + 1) * width]
        if skip_unused and (x[0] < 0 or y[0] < 0):
            continue
        same += len(set(x) & set(y))
        total += width
    return same, total


def share(a, b, p):
    n = TOKENS[p]
    ta = load(f"{a}.p{p}.topk", "I")[DENSE * n * K:]
    tb = load(f"{b}.p{p}.topk", "I")[DENSE * n * K:]
    s, t = common(ta, tb, K, False)
    ps, pt = common(load(f"{a}.p{p}.pools", "i"), load(f"{b}.p{p}.pools", "i"), TOP, True)
    print(f"prompt {p}: experts {s}/{t} ({100 * s / t:.3f}%), DSA pools {ps}/{pt} ({100 * ps / max(pt, 1):.3f}%)")


for p in TOKENS:
    share(sys.argv[1], sys.argv[2], p)
