#!/usr/bin/env python3
"""Checks that `--record` costs no speed and changes no result (TASK-16).

Runs `tfswarm ... --context N --tokens F --generate M` once with `--save` as
the reference, then alternates runs without and with `--record`, each with
`--compare` against the reference, so every run must give the same tokens
and logits bit for bit. Reports the median decode speed of each, and checks
that every recording holds one line per engine step, with nothing dropped.
--peer starts rank 1 before each run of a split. Standard library only.

Usage:
  python3 tools/ref/check_record.py --cmd "zig-out/bin/tfswarm local MODEL --gpu" [--peer "..."] [--context 512] [--generate 256] [--reps 3]
"""
import argparse
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time


def run(args, extra, peer):
    p = subprocess.Popen(shlex.split(peer), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if peer else None
    if p:
        time.sleep(args.peer_wait)
    r = subprocess.run(shlex.split(args.cmd) + extra, capture_output=True, text=True)
    if p:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit("tfswarm failed")
    return r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="tfswarm and its arguments, without --context/--tokens/--generate")
    ap.add_argument("--peer", default="", help="a command that starts rank 1 for each run")
    ap.add_argument("--peer-wait", type=float, default=0.0)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--generate", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--keep", help="keep the last recording here")
    args = ap.parse_args()

    rate = re.compile(r"decode [\d.]+ s per token \(([\d.]+) tokens/s\)")
    with tempfile.TemporaryDirectory() as d:
        toks, ref, rec = (os.path.join(d, n) for n in ("prompt.tokens", "ref.rec", "trace.jsonl"))
        rng = random.Random(16)
        with open(toks, "wb") as f:
            f.write(b"".join(rng.randrange(100, 50000).to_bytes(4, "little") for _ in range(args.context)))
        base = ["--context", str(args.context), "--tokens", toks, "--generate", str(args.generate)]
        run(args, base + ["--save", ref], args.peer)
        speeds = {"off": [], "on": []}
        ok = True
        for rep in range(args.reps):
            for mode in ("off", "on"):
                out = run(args, base + ["--compare", ref] + (["--record", rec] if mode == "on" else []), args.peer)
                same = "PASS: all" in out
                speed = float(rate.search(out).group(1))
                speeds[mode].append(speed)
                line = f"rep {rep}, recording {mode:3}: {speed:.2f} tokens/s, tokens and logits {'identical to the reference' if same else 'DIFFER'}"
                if mode == "on":
                    lines = [json.loads(l) for l in open(rec, encoding="utf-8")]
                    steps = [l for l in lines if l.get("t") == "step"]
                    end = lines[-1]
                    # Prefill in chunks, then one step per generated token after the first.
                    seqs = [s["seq"] for s in steps]
                    whole = lines[0]["t"] == "hello" and end["t"] == "end" and end["dropped"] == 0 and seqs == list(range(len(seqs)))
                    decode = sum(1 for s in steps if s["n"] == 1)
                    whole = whole and sum(s["n"] for s in steps) == args.context + args.generate - 1 and decode >= args.generate - 1
                    line += f"; recording: {len(steps)} steps, {os.path.getsize(rec) / 1e6:.2f} MB, {'whole' if whole else 'INCOMPLETE'}"
                    ok = ok and whole
                    if args.keep:
                        with open(rec, "rb") as src, open(args.keep, "wb") as dst:
                            dst.write(src.read())
                ok = ok and same
                print(line, flush=True)
    off, on = statistics.median(speeds["off"]), statistics.median(speeds["on"])
    change = 100 * (on / off - 1)
    ok = ok and change > -2
    print(f"median decode: {off:.2f} tokens/s without recording, {on:.2f} with ({change:+.1f}%)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
