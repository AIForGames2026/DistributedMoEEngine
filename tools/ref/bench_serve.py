#!/usr/bin/env python3
"""Times `tfswarm ... --serve` against `tfswarm` itself (TASK-30), in two steps.

  server: sends each prompt greedily as a streamed chat request, with no
          viewer and with a viewer reading the live diagram's feed, and
          records time to first token and decode speed as the client sees
          them and as the server reports them. A reply whose first tokens
          show no text (Qwen3 opens with <think> and a newline) reaches the
          client that many decode steps later; --lead N says how many. The server needs --no-reuse so
          every request prefills its whole prompt, like olmoe. Run it on the
          server's machine.
  direct: runs `tfswarm MODE MODEL --context N --tokens F --generate M` on the
          same prompt tokens, M being what the server generated, and records
          what tfswarm reports. --peer starts rank 1 before each run of a split.

Standard library only.

Usage:
  python3 tools/ref/bench_serve.py server --base http://127.0.0.1:8091 --out serve.json [--reps 3] [--max-tokens 128]
  python3 tools/ref/bench_serve.py direct --results serve.json --cmd "zig-out/bin/tfswarm local MODEL --gpu" [--peer "..."] [--reps 3]
"""
import argparse
import http.client
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_live import Viewer  # noqa: E402

LONG = " ".join(f"Note {i}: the relay at station {i} reported nominal power, a clear channel, and {i % 7} queued messages." for i in range(1, 61))
PROMPTS = {
    "short": [{"role": "user", "content": "Why is the sky blue?"}],
    "medium": [{"role": "system", "content": "You are a careful assistant. " * 12},
               {"role": "user", "content": "Explain how a mixture-of-experts language model decides which experts run for each token, "
                                           "and why that makes it cheaper to run than a dense model of the same size. " * 3}],
    "long": [{"role": "user", "content": LONG + "\n\nSummarize the station reports above in three sentences."}],
}


def stream(base, messages, max_tokens, lead=0):
    """One streamed greedy request; returns client-side times and the server's own stats."""
    host = base.split("//")[1]
    conn = http.client.HTTPConnection(host, timeout=600)
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": True,
                       "return_tokens": True, "stream_options": {"include_usage": True}})
    t0 = time.monotonic()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    first = last = None
    extra = None
    buf = b""
    while True:
        data = r.read1(1 << 16)
        if not data:
            break
        buf += data
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            line = event.decode()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("choices"):
                d = chunk["choices"][0]["delta"]
                if d.get("content") or d.get("reasoning_content"):
                    now = time.monotonic()
                    first = first if first is not None else now
                    last = now
            if "tfswarm" in chunk:
                extra = chunk["tfswarm"]
    conn.close()
    n = len(extra["completion_ids"])
    return {
        "client_ttft_ms": (first - t0) * 1000,
        # The first and last pieces of text bound the decode steps after the `lead` tokens that show none.
        "client_decode_tps": (n - 1 - lead) / (last - first) if last > first else 0.0,
        "server_ttft_ms": extra["ttft_ms"],
        "server_decode_tps": extra["decode_tokens_per_s"],
        "cached": extra["cached_tokens"],
        "prompt_ids": extra["prompt_ids"],
        "completion_ids": extra["completion_ids"],
    }


def server(args):
    stream(args.base, PROMPTS["short"], 4)
    out = {"base": args.base, "max_tokens": args.max_tokens, "lead": args.lead, "runs": []}
    for name, msgs in PROMPTS.items():
        for viewer in ("none", "reading"):
            for rep in range(args.reps):
                v = Viewer(args.base) if viewer == "reading" else None
                if v:
                    time.sleep(0.3)
                r = stream(args.base, msgs, args.max_tokens, args.lead)
                if v:
                    time.sleep(0.3)
                    got = len(v.frames)
                    v.close()
                if r["cached"] != 0:
                    raise SystemExit("the server reused a cached prefix; start it with --no-reuse")
                r.update(prompt=name, viewer=viewer, rep=rep)
                out["runs"].append(r)
                print(f"{name:6} viewer {viewer:7} rep {rep}: {len(r['prompt_ids'])} prompt tokens, {len(r['completion_ids'])} generated; "
                      f"first token {r['client_ttft_ms']:.0f} ms at the client ({r['server_ttft_ms']:.0f} in the engine), "
                      f"decode {r['client_decode_tps']:.2f} tokens/s at the client ({r['server_decode_tps']:.2f} in the engine)"
                      f"{f'; viewer got {got} frames' if v else ''}")
    same = True
    for name in PROMPTS:
        ids = [r["completion_ids"] for r in out["runs"] if r["prompt"] == name]
        same = same and all(i == ids[0] for i in ids)
    print(f"tokens identical across reps and viewers: {same}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    summarize(out)


def direct(args):
    with open(args.results, encoding="utf-8") as f:
        out = json.load(f)
    pat = re.compile(r"time to first token ([\d.]+) s .*?decode [\d.]+ s per token \(([\d.]+) tokens/s\)")
    out["direct"] = []
    with tempfile.TemporaryDirectory() as d:
        for name in PROMPTS:
            r = next(x for x in out["runs"] if x["prompt"] == name)
            prompt, n = r["prompt_ids"], len(r["completion_ids"])
            toks = os.path.join(d, f"{name}.tokens")
            with open(toks, "wb") as f:
                f.write(b"".join(i.to_bytes(4, "little") for i in prompt))
            for rep in range(args.reps):
                peer = subprocess.Popen(shlex.split(args.peer), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if args.peer else None
                if peer:
                    time.sleep(args.peer_wait)
                cmd = shlex.split(args.cmd) + ["--context", str(len(prompt)), "--tokens", toks, "--generate", str(n)]
                run = subprocess.run(cmd, capture_output=True, text=True)
                if peer:
                    try:
                        peer.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        peer.kill()
                m = pat.search(run.stderr)
                if run.returncode != 0 or not m:
                    print(run.stderr[-2000:], file=sys.stderr)
                    raise SystemExit(f"tfswarm failed on {name}")
                x = {"prompt": name, "rep": rep, "ttft_ms": float(m.group(1)) * 1000, "decode_tps": float(m.group(2))}
                out["direct"].append(x)
                print(f"{name:6} direct rep {rep}: {len(prompt)} prompt tokens, {n} generated; first token {x['ttft_ms']:.0f} ms, decode {x['decode_tps']:.2f} tokens/s")
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(out, f)
    summarize(out)


def summarize(out):
    med = statistics.median
    print("\n| Prompt | Prompt tokens | Generated | tfswarm directly | Server, engine | Server, client | Server + viewer, engine | Server + viewer, client |")
    print("|---|---|---|---|---|---|---|---|")
    for name in PROMPTS:
        runs = [r for r in out["runs"] if r["prompt"] == name]
        cells = []
        dr = [x for x in out.get("direct", []) if x["prompt"] == name]
        cells.append(f"{med(x['ttft_ms'] for x in dr):.0f} ms, {med(x['decode_tps'] for x in dr):.2f} tokens/s" if dr else "")
        for viewer in ("none", "reading"):
            rs = [r for r in runs if r["viewer"] == viewer]
            cells.append(f"{med(r['server_ttft_ms'] for r in rs):.0f} ms, {med(r['server_decode_tps'] for r in rs):.2f} tokens/s")
            cells.append(f"{med(r['client_ttft_ms'] for r in rs):.0f} ms, {med(r['client_decode_tps'] for r in rs):.2f} tokens/s")
        print(f"| {name} | {len(runs[0]['prompt_ids'])} | {len(runs[0]['completion_ids'])} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    s = sub.add_parser("server")
    s.add_argument("--base", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--lead", type=int, default=0, help="tokens each reply opens with that show no text (2 for Qwen3's <think> and newline)")
    d = sub.add_parser("direct")
    d.add_argument("--results", required=True)
    d.add_argument("--cmd", required=True, help="tfswarm and its arguments, without --context/--tokens/--generate")
    d.add_argument("--peer", default="", help="a command that starts rank 1 for each run")
    d.add_argument("--peer-wait", type=float, default=5.0, help="seconds to let rank 1 load before starting rank 0")
    d.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    server(args) if args.cmd_name == "server" else direct(args)


if __name__ == "__main__":
    main()
