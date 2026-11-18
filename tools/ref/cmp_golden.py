#!/usr/bin/env python3
"""Compares two .tfref goldens of the same prompts, such as one reference run with
different thread counts (TASK-31, TASK-32): the expert choices and DSA pools or QSA
blocks both kept, and the logits.

    python cmp_golden.py A.tfref B.tfref
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import parse  # noqa: E402


def common(a, b):
    """Entries the two index tensors share, row by row (order within a row ignored)."""
    same = 0
    for x, y in zip(a.tolist(), b.tolist()):
        if x[0] < 0 or y[0] < 0:
            continue
        same += len(set(x) & set(y))
    return same


def main():
    with open(sys.argv[1], "rb") as f:
        a = parse(f.read())
    with open(sys.argv[2], "rb") as f:
        b = parse(f.read())
    prompts = int(a["config"][0])
    layers = int(a["config"][1])
    for p in range(prompts):
        if not torch.equal(a[f"prompt{p}.tokens"], b[f"prompt{p}.tokens"]):
            sys.exit(f"prompt {p}: the two goldens have different tokens")
        es = et = bs = bt = 0
        for kind in ("dsa_pools", "qsa_blocks"):
            for l in range(layers):
                name = f"prompt{p}.layer{l}.{kind}"
                if name in a and name in b:
                    s = common(a[name], b[name])
                    t = int((a[name][:, 0] >= 0).sum()) * a[name].shape[1]
                    bs += s
                    bt += t
                    if s != t:
                        print(f"prompt {p}, layer {l}: {kind} {s}/{t} same")
        for l in range(layers):
            name = f"prompt{p}.layer{l}.topk_idx"
            if name in a and name in b:
                es += common(a[name], b[name])
                et += a[name].numel()
        la = a[f"prompt{p}.logits"]
        lb = b[f"prompt{p}.logits"]
        err = float((la - lb).abs().max())
        top1 = int((la.argmax(-1) == lb.argmax(-1)).sum())
        blocks = f", pools or blocks {bs}/{bt} same ({100 * bs / bt:.3f}%)" if bt else ""
        print(f"prompt {p} ({a[f'prompt{p}.tokens'].numel()} tokens): experts {es}/{et} same "
              f"({100 * es / max(et, 1):.3f}%){blocks}, logits max err {err:.3e}, top-1 {top1}/{la.shape[0]}")


if __name__ == "__main__":
    main()
