#!/usr/bin/env python3
"""Checks the engine's chat formatting (TASK-30) against transformers'
`apply_chat_template`: renders and tokenizes a set of conversations both ways
and compares the prompt token ids.

Usage:
  python3 tools/ref/check_chat.py --hf-dir DIR --tftok zig-out/bin/tftok \
      --source MODEL.gguf|tokenizer.json [--config tokenizer_config.json]

DIR holds the model's tokenizer.json and tokenizer_config.json. With
--gguf-template MODEL.gguf, transformers renders with that file's template,
which may be an older revision than DIR's.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

from transformers import AutoTokenizer


def u(c):
    return {"role": "user", "content": c}


def a(c, reasoning=None):
    m = {"role": "assistant", "content": c}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def s(c):
    return {"role": "system", "content": c}


CONVERSATIONS = [
    [u("Hello!")],
    [s("You are a helpful assistant."), u("What is the capital of France?")],
    [u("Hi"), a("Hello! How can I help?"), u("Tell me a joke.")],
    [s("Be brief."), u("One"), a("Two"), u("Three"), a("Four"), u("Five")],
    [u("Hi"), a("<think>\nThe user greets me.\n</think>\n\nHello!"), u("And again?")],
    [u("Q"), a("<think>\n\nr1\n\n</think>\n\n\nA1")],
    [u("Q"), a("plain answer")],
    [u("Q"), a("Answer", reasoning="\nkept reasoning\n")],
    [u("Q1"), a("A1", reasoning="dropped"), u("Q2"), a("A2", reasoning="kept")],
    [u("Q"), a("x</think>y</think>z")],
    [u("Q"), a("<think>a<think>b</think>c")],
    [u("first"), s("a system turn in the middle"), u("second")],
    [u("")],
    [u("Typed specials: <|im_start|> <|endoftext|> <think> <|user|>")],
    [u("café, 中文, \U0001F600, tabs\tand\nnewlines\n\n")],
    [u("<tool_response>\nnot a real query\n</tool_response>"), a("ok")],
    [u("real query"), a("A", reasoning="r"), u("<tool_response>\nx\n</tool_response>")],
    [u("Q"), a("A", reasoning="r"), u("see <tool_response>x</tool_response> here"), a("B", reasoning="r2")],
]

SETTINGS = [
    {"add_generation_prompt": True},
    {"add_generation_prompt": False},
    {"add_generation_prompt": True, "enable_thinking": False},
    {"add_generation_prompt": True, "enable_thinking": True},
    {"add_generation_prompt": True, "reasoning_effort": "low"},
    {"add_generation_prompt": True, "reasoning_effort": "medium"},
]


def gguf_template(path):
    """The `tokenizer.chat_template` string from a GGUF file's metadata."""
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        def string():
            (n,) = rd("Q")
            return f.read(n).decode("utf-8")

        def skip(t):
            if t == 8:
                string()
            elif t == 9:
                (et,) = rd("I")
                (n,) = rd("Q")
                for _ in range(n):
                    skip(et)
            else:
                rd(sizes[t])

        f.seek(16)
        (n_kv,) = rd("Q")
        for _ in range(n_kv):
            key = string()
            (t,) = rd("I")
            if key == "tokenizer.chat_template" and t == 8:
                return string()
            skip(t)
    raise SystemExit(f"{path} has no chat template")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--tftok", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config")
    ap.add_argument("--gguf-template", help="render with this GGUF's template instead of DIR's")
    args = ap.parse_args()

    hf = AutoTokenizer.from_pretrained(args.hf_dir)
    if args.gguf_template:
        hf.chat_template = gguf_template(args.gguf_template)
    cases = [(c, st) for c in CONVERSATIONS for st in SETTINGS]
    with tempfile.TemporaryDirectory() as d:
        cmd = [os.path.abspath(args.tftok), args.source]
        if args.config:
            cmd += ["--config", args.config]
        for i, (conv, st) in enumerate(cases):
            p = os.path.join(d, f"{i}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"messages": conv, **st}, f)
            cmd += ["--chat", p]
        run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        sys.exit(1)
    lines = run.stdout.split("\n")

    bad = 0
    refused = 0
    for (conv, st), line in zip(cases, lines):
        try:
            want = hf.apply_chat_template(conv, tokenize=True, **st)
            # transformers 5 returns the encoding as a mapping.
            want = list(want["input_ids"] if hasattr(want, "keys") else want)
        except Exception as e:
            # Templates may refuse a conversation (Qwen3.8's: a late system turn, no user query); so must the engine.
            if not line.startswith("refused"):
                bad += 1
                print(f"MISMATCH {st} {conv}: transformers refused ({e}), the engine did not")
            else:
                refused += 1
            continue
        if line.startswith("refused"):
            bad += 1
            print(f"MISMATCH {st} {conv}: the engine refused ({line}), transformers did not")
            continue
        got = [int(x) for x in line.split()]
        if got != want:
            bad += 1
            if bad <= 5:
                print(f"MISMATCH {st} {conv}")
                print(f"  hf:     {hf.decode(want)!r}")
                print(f"  engine: {hf.decode(got)!r}")
    print(f"{len(cases) - bad}/{len(cases)} prompts identical ({refused} refused by both)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
