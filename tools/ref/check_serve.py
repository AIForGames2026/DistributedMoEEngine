#!/usr/bin/env python3
"""Checks `tfswarm ... --serve` (TASK-30) against `tfswarm` itself, in two steps.

  requests: sends each test conversation greedily through curl and the
            `openai` package, streamed and not, checks that all four give the
            same tokens and text, and saves the prompt and completion token ids.
            Run it where the server is reachable; the server needs --no-reuse
            so every request runs from an empty cache, like olmoe.
  compare:  feeds each saved prompt to `tfswarm local --context N --tokens F
            --generate M --save R` and checks that it generates the same tokens.
            Run it on the engine machine; it needs only the standard library.

Usage:
  python3 tools/ref/check_serve.py requests --base http://127.0.0.1:8091/v1 --out serve.json
  python3 tools/ref/check_serve.py compare --results serve.json --tfswarm zig-out/bin/tfswarm --model MODEL.gguf [--extra "--gpu"]
"""
import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile

CONVERSATIONS = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "system", "content": "You answer in one short sentence."},
     {"role": "user", "content": "Why is the sky blue?"}],
    [{"role": "user", "content": "Write a Python function that reverses a string."}],
    [{"role": "user", "content": "Hi!"},
     {"role": "assistant", "content": "Hello! How can I help you today?"},
     {"role": "user", "content": "Name three mixture-of-experts language models."}],
]


def via_curl(base, body):
    out = subprocess.run(["curl", "-sN", base + "/chat/completions", "-H", "Content-Type: application/json",
                          "-d", json.dumps(body)], capture_output=True, check=True).stdout.decode("utf-8")
    if not body.get("stream"):
        r = json.loads(out)
        if "error" in r:
            raise SystemExit(f"server error: {r['error']}")
        m = r["choices"][0]["message"]
        return (m.get("reasoning_content", ""), m["content"]), r["tfswarm"], r["choices"][0]["finish_reason"]
    reasoning, text, extra, finish = "", "", None, None
    for line in out.split("\n"):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk["choices"]:
            delta = chunk["choices"][0]["delta"]
            reasoning += delta.get("reasoning_content") or ""
            text += delta.get("content") or ""
            finish = chunk["choices"][0]["finish_reason"] or finish
        if "tfswarm" in chunk:
            extra = chunk["tfswarm"]
    return (reasoning, text), extra, finish


def via_openai(client, body):
    extra_body = {"return_tokens": True}
    if not body.get("stream"):
        r = client.chat.completions.create(model=body["model"], messages=body["messages"], temperature=0,
                                           max_tokens=body["max_tokens"], extra_body=extra_body)
        m = r.choices[0].message
        return ((m.model_extra or {}).get("reasoning_content", ""), m.content), r.model_extra["tfswarm"], r.choices[0].finish_reason
    reasoning, text, extra, finish = "", "", None, None
    for chunk in client.chat.completions.create(model=body["model"], messages=body["messages"], temperature=0,
                                                max_tokens=body["max_tokens"], stream=True,
                                                stream_options={"include_usage": True}, extra_body=extra_body):
        if chunk.choices:
            delta = chunk.choices[0].delta
            reasoning += (delta.model_extra or {}).get("reasoning_content") or ""
            text += delta.content or ""
            finish = chunk.choices[0].finish_reason or finish
        if chunk.model_extra and "tfswarm" in chunk.model_extra:
            extra = chunk.model_extra["tfswarm"]
    return (reasoning, text), extra, finish


def requests(args):
    from openai import OpenAI

    client = OpenAI(base_url=args.base, api_key="none")
    model = client.models.list().data[0].id
    results, ok = [], True
    for i, conv in enumerate(CONVERSATIONS):
        runs = []
        for stream in (False, True):
            body = {"model": model, "messages": conv, "temperature": 0, "max_tokens": args.max_tokens,
                    "stream": stream, "return_tokens": True}
            runs.append(("curl", stream) + via_curl(args.base, body))
            runs.append(("openai", stream) + via_openai(client, body))
        first = runs[0]
        for name, stream, text, extra, finish in runs:
            if extra["cached_tokens"] != 0:
                raise SystemExit("the server reused a cached prefix; start it with --no-reuse")
            same = text == first[2] and extra["completion_ids"] == first[3]["completion_ids"] and extra["prompt_ids"] == first[3]["prompt_ids"]
            ok = ok and same
            print(f"conversation {i}: {name:6} {'stream' if stream else 'whole ':6} {len(extra['completion_ids'])} tokens, "
                  f"finish {finish}, first token {extra['ttft_ms']:.0f} ms, {extra['decode_tokens_per_s']:.1f} tokens/s"
                  f"{'' if same else '  DIFFERS from curl, whole'}")
        reasoning, content = first[2]
        results.append({"prompt_ids": first[3]["prompt_ids"], "completion_ids": first[3]["completion_ids"], "reasoning": reasoning, "content": content})
        if reasoning:
            print(f"  reasoning: {reasoning[:90]!r}")
        print(f"  content:   {content[:90]!r}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "results": results}, f)
    print(f"{'PASS' if ok else 'FAIL'}: curl and openai, streamed and not, {'agree' if ok else 'disagree'} on all {len(results)} conversations; wrote {args.out}")
    sys.exit(0 if ok else 1)


def compare(args):
    with open(args.results, encoding="utf-8") as f:
        saved = json.load(f)["results"]
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for i, r in enumerate(saved):
            prompt, want = r["prompt_ids"], r["completion_ids"]
            toks, rec = os.path.join(d, f"{i}.tokens"), os.path.join(d, f"{i}.rec")
            with open(toks, "wb") as f:
                f.write(struct.pack(f"<{len(prompt)}I", *prompt))
            cmd = [args.tfswarm, "local", args.model, "--context", str(len(prompt)), "--tokens", toks,
                   "--generate", str(len(want)), "--save", rec] + shlex.split(args.extra)
            run = subprocess.run(cmd, capture_output=True, text=True)
            if run.returncode != 0:
                print(run.stderr, file=sys.stderr)
                raise SystemExit(f"tfswarm failed on conversation {i}")
            with open(rec, "rb") as f:
                data = f.read()
            n = struct.unpack("<Q", data[8:16])[0]
            got = list(struct.unpack(f"<{n}I", data[16:16 + 4 * n]))
            same = got == want
            ok = ok and same
            print(f"conversation {i}: {len(prompt)} prompt tokens, {len(want)} generated: {'same as tfswarm' if same else 'DIFFERS'}")
            if not same:
                at = next(j for j, (a, b) in enumerate(zip(got, want)) if a != b)
                print(f"  first difference at token {at}: tfswarm {got[at:at + 5]}, server {want[at:at + 5]}")
    print(f"{'PASS' if ok else 'FAIL'}: the server's tokens {'match' if ok else 'differ from'} tfswarm's for the same prompt tokens")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("requests")
    r.add_argument("--base", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=48)
    c = sub.add_parser("compare")
    c.add_argument("--results", required=True)
    c.add_argument("--tfswarm", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--extra", default="")
    args = ap.parse_args()
    requests(args) if args.cmd == "requests" else compare(args)


if __name__ == "__main__":
    main()
