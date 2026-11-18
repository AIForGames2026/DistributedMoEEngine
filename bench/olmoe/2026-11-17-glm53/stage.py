#!/usr/bin/env python3
"""TASK-32: stages one expert rank's share of GLM-5.3-Flash's GGUF on another Spark, as a sparse
file of the full length holding the header, every non-expert tensor, and this rank's experts
(an even split: rank r of n owns experts r*E/n to (r+1)*E/n - 1 of every layer). Bytes go over
the direct cable by ssh; run on gx10-02:

  stage.py PART2.gguf HOST RANK RANKS
  stage.py PART2.gguf HOST --add FIRST LAST

--add copies only experts FIRST to LAST of every layer into a share already staged on HOST.
"""
import os
import subprocess
import sys

import gguf


def main():
    path, host = sys.argv[1], sys.argv[2]
    add = sys.argv[3] == "--add"
    if add:
        lo, hi = int(sys.argv[4]), int(sys.argv[5]) + 1
    else:
        rank, ranks = int(sys.argv[3]), int(sys.argv[4])
    r = gguf.GGUFReader(path)
    size = os.path.getsize(path)
    spans = [] if add else [(0, int(r.tensors[0].data_offset))]
    for t in r.tensors:
        off, n = int(t.data_offset), int(t.n_bytes)
        if "_exps." in t.name:
            experts = int(t.shape[-1])
            per = n // experts
            first, last = (lo, hi) if add else (rank * experts // ranks, (rank + 1) * experts // ranks)
            spans.append((off + first * per, (last - first) * per))
        elif not add:
            spans.append((off, n))
    spans.sort()
    merged = []
    for off, n in spans:
        if merged and off <= merged[-1][0] + merged[-1][1]:
            end = max(merged[-1][0] + merged[-1][1], off + n)
            merged[-1] = (merged[-1][0], end - merged[-1][0])
        else:
            merged.append((off, n))
    total = sum(n for _, n in merged)
    dst = "~/lukehinojosa/models/glm53-flash/" + os.path.basename(path)
    what = f"experts {lo} to {hi - 1} added" if add else f"rank {rank} of {ranks}"
    print(f"{host}: {what}, {len(merged)} spans, {total / 1e9:.2f} GB of {size / 1e9:.2f} GB", flush=True)
    if add:
        subprocess.run(["ssh", host, f"test $(stat -c %s {dst}) = {size}"], check=True)
    else:
        subprocess.run(["ssh", host, f"mkdir -p ~/lukehinojosa/models/glm53-flash && rm -f {dst} && truncate -s {size} {dst}"], check=True)
    sent = 0
    for off, n in merged:
        # Whole MiB blocks where possible; dd's byte flags cover the rest.
        src = subprocess.Popen(["dd", f"if={path}", "bs=4M", "iflag=skip_bytes,count_bytes", f"skip={off}", f"count={n}", "status=none"], stdout=subprocess.PIPE)
        subprocess.run(["ssh", host, f"dd of={dst} bs=4M oflag=seek_bytes seek={off} conv=notrunc status=none"], stdin=src.stdout, check=True)
        src.stdout.close()
        if src.wait() != 0:
            sys.exit(f"reading {n} bytes at {off} failed")
        sent += n
        print(f"\r{host}: {sent / 1e9:.1f} of {total / 1e9:.1f} GB", end="", flush=True)
    print(f"\n{host}: staged {dst}", flush=True)


if __name__ == "__main__":
    main()
