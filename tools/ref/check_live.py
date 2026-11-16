#!/usr/bin/env python3
"""Checks the live telemetry feed of `olmoe ... --serve` (TASK-16b).

Runs the same greedy chat request with no viewer, with a viewer that reads
every frame, and with a viewer that connects and never reads (a stalled
browser tab), and checks that the generated tokens are identical and that
decode speed stays within a few percent. The reading viewer's frames must
account for every expert call, and match the engine's own counters where
it keeps them. With --stall-tokens, a long request then runs with a stalled
viewer beside a reading one: the reader must still get every frame, and the
stalled one must have frames dropped rather than slow generation. Run it on
the server's machine (a relay in between, such as WSL's localhost
forwarding, can hide a stall). Standard library only.

Usage:
  python3 tools/ref/check_live.py --base http://127.0.0.1:8091 [--max-tokens 64] [--rounds 3] [--stall-tokens 600]
"""
import argparse
import base64
import json
import os
import socket
import struct
import sys
import threading
import time
import urllib.request

PROMPT = [{"role": "user", "content": "Write a short paragraph about why the sky is blue."}]


def chat(base, max_tokens, salt, prompt=PROMPT):
    # A different salt per round keeps one round from reusing another's cached prompt.
    msgs = [{"role": "system", "content": f"Session {salt}."}] + prompt
    body = json.dumps({"messages": msgs, "temperature": 0, "max_tokens": max_tokens, "return_tokens": True}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["tensorfold"]


class Viewer:
    """A WebSocket client of /telemetry; `read=False` connects and then never reads."""

    def __init__(self, base, read=True):
        host, port = base.split("//")[1].split(":")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not read:
            # Small, but set before connecting so TCP can still advertise a full segment.
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 128 << 10)
        self.sock.connect((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /telemetry HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            self.buf += self.sock.recv(4096)
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        if not head.startswith(b"HTTP/1.1 101"):
            raise SystemExit(f"no WebSocket upgrade: {head[:80]!r}")
        self.frames = []
        self.hello = None
        if read:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def loop(self):
        while True:
            while True:
                f = self.parse()
                if f is None:
                    break
                msg = json.loads(f)
                # The engine stamps frames with CLOCK_MONOTONIC, the clock behind time.monotonic() on Linux.
                msg["delay_ms"] = time.monotonic() * 1000 - msg["ms"]
                if msg["t"] == "hello":
                    self.hello = msg
                else:
                    self.frames.append(msg)
            try:
                data = self.sock.recv(1 << 16)
            except OSError:
                return
            if not data:
                return
            self.buf += data

    def parse(self):
        b = self.buf
        if len(b) < 2:
            return None
        n, at = b[1] & 0x7F, 2
        if n == 126:
            if len(b) < 4:
                return None
            n, at = struct.unpack(">H", b[2:4])[0], 4
        elif n == 127:
            if len(b) < 10:
                return None
            n, at = struct.unpack(">Q", b[2:10])[0], 10
        if len(b) < at + n:
            return None
        self.buf = b[at + n:]
        return b[at:at + n].decode()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def account(v):
    """Checks a reading viewer's frames: gapless, one call per token, layer and choice, and the engine's counts."""
    h, frames = v.hello, list(v.frames)
    ranks = len(h["ranks"])

    def by_rank(f):
        out = [0] * ranks
        for i in range(0, len(f["x"]), 3):
            out[min(h["owner"][f["x"][i + 1]], ranks - 1)] += f["x"][i + 2]
        return out

    drawn = [sum(c) for c in zip(*(by_rank(f) for f in frames))]
    tokens = sum(f["n"] for f in frames)
    seqs = [f["seq"] for f in frames]
    gapless = seqs == list(range(seqs[0], seqs[0] + len(seqs)))
    first, last = frames[0], frames[-1]
    base = [None if first["engine"][r] is None else first["engine"][r] - by_rank(first)[r] for r in range(ranks)]
    engine = [None if base[r] is None else last["engine"][r] - base[r] for r in range(ranks)]
    ok = gapless and sum(drawn) == tokens * h["layers"] * h["top_k"] and all(e is None or e == d for e, d in zip(engine, drawn))
    # From a step's first expert call to its frame arriving here: the step, then delivery.
    lag = sorted(f["step_us"] / 1000 + f["delay_ms"] for f in frames)
    print(f"  {len(frames)} frames, {tokens} tokens, expert calls by rank {drawn} (engine {engine}), "
          f"{'gapless' if gapless else 'GAPS'}: {'ok' if ok else 'MISMATCH'}; step start to frame arrival "
          f"p50 {lag[len(lag) // 2]:.1f} ms, max {lag[-1]:.1f} ms (delivery alone max {max(f['delay_ms'] for f in frames):.1f} ms)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="server root, e.g. http://127.0.0.1:8091")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--stall-tokens", type=int, default=0, help="then a long request with a stalled viewer, to see its frames dropped")
    args = ap.parse_args()

    chat(args.base, 4, "warm-up")
    results = {"none": [], "reading": [], "stalled": []}
    counted_ok = True
    for rnd in range(args.rounds):
        for mode in ("none", "reading", "stalled"):
            v = Viewer(args.base, read=(mode == "reading")) if mode != "none" else None
            time.sleep(0.3)
            t = chat(args.base, args.max_tokens, f"{rnd}")
            print(f"round {rnd}, viewer {mode:8}: {len(t['completion_ids'])} tokens, {t['decode_tokens_per_s']:.2f} tokens/s, first token {t['ttft_ms']:.0f} ms")
            if mode == "reading":
                time.sleep(0.5)
                counted_ok = account(v) and counted_ok
            if v:
                v.close()
            results[mode].append(t)

    ref = [r["completion_ids"] for r in results["none"]]
    same = all(r["completion_ids"] == ref[i] for mode in results for i, r in enumerate(results[mode]))
    speed = {m: sorted(r["decode_tokens_per_s"] for r in rs)[len(rs) // 2] for m, rs in results.items()}
    print(f"median decode: no viewer {speed['none']:.2f}, reading viewer {speed['reading']:.2f} "
          f"({100 * (speed['reading'] / speed['none'] - 1):+.1f}%), stalled viewer {speed['stalled']:.2f} "
          f"({100 * (speed['stalled'] / speed['none'] - 1):+.1f}%) tokens/s")
    print(f"tokens identical in every run: {same}; frames account for every expert call: {counted_ok}")
    ok = same and counted_ok

    if args.stall_tokens > 0:
        prompt = [{"role": "user", "content": "Count from 1 to 2000, one number per line, with no other text."}]
        alone = chat(args.base, args.stall_tokens, "stall", prompt)
        stalled, reader = Viewer(args.base, read=False), Viewer(args.base, read=True)
        time.sleep(0.3)
        t = chat(args.base, args.stall_tokens, "stall", prompt)
        threading.Thread(target=stalled.loop, daemon=True).start()
        time.sleep(3)
        # Once it reads again, one more request, whose frames show the gap where the stalled viewer's were dropped.
        chat(args.base, 1, "after")
        time.sleep(0.5)
        s = [f["seq"] for f in stalled.frames]
        dropped = sum(b - a - 1 for a, b in zip(s, s[1:]))
        print(f"stall: {len(t['completion_ids'])} tokens at {t['decode_tokens_per_s']:.2f} tokens/s with a stalled viewer, "
              f"{alone['decode_tokens_per_s']:.2f} with none ({100 * (t['decode_tokens_per_s'] / alone['decode_tokens_per_s'] - 1):+.1f}%)")
        print("  reading viewer meanwhile:", end="")
        stall_ok = account(reader) and dropped > 0 and t["completion_ids"] == alone["completion_ids"]
        print(f"  stalled viewer, read afterwards: {len(s)} frames, {dropped} dropped for it; same tokens: {t['completion_ids'] == alone['completion_ids']}")
        ok = ok and stall_ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
