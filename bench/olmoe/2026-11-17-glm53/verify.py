#!/usr/bin/env python3
"""TASK-32: checks a staged share against gx10-02's GGUF without holding it in memory. Hashes
the spans stage.py copied in 4 MiB reads, dropping each from the page cache, so it is safe
on a Spark whose lab server is running. On gx10-02:

  verify.py spans PART2.gguf RANK RANKS > spans.txt
  verify.py hash FILE spans.txt
"""
import hashlib
import os
import sys

CHUNK = 4 << 20
CHECKPOINT = 4 << 30


def spans(path, rank, ranks):
    import gguf
    r = gguf.GGUFReader(path)
    out = [(0, int(r.tensors[0].data_offset))]
    for t in r.tensors:
        off, n = int(t.data_offset), int(t.n_bytes)
        if "_exps." in t.name:
            experts = int(t.shape[-1])
            per = n // experts
            first, last = rank * experts // ranks, (rank + 1) * experts // ranks
            out.append((off + first * per, (last - first) * per))
        else:
            out.append((off, n))
    out.sort()
    merged = []
    for off, n in out:
        if merged and off <= merged[-1][0] + merged[-1][1]:
            end = max(merged[-1][0] + merged[-1][1], off + n)
            merged[-1] = (merged[-1][0], end - merged[-1][0])
        else:
            merged.append((off, n))
    for off, n in merged:
        print(off, n)


def digest(path, spanfile):
    todo = [tuple(map(int, line.split())) for line in open(spanfile) if line.strip()]
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    done = 0
    mark = CHECKPOINT
    for off, n in todo:
        at = off
        while at < off + n:
            want = min(CHUNK, off + n - at)
            b = os.pread(fd, want, at)
            if len(b) != want:
                sys.exit(f"short read at {at}")
            h.update(b)
            os.posix_fadvise(fd, at, want, os.POSIX_FADV_DONTNEED)
            at += want
            done += want
            if done >= mark:
                print(f"{done >> 30} GiB {h.copy().hexdigest()[:16]}", flush=True)
                mark += CHECKPOINT
    print(f"total {done} {h.hexdigest()}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "spans":
        spans(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        digest(sys.argv[2], sys.argv[3])
