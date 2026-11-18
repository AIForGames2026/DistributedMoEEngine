#!/usr/bin/env python3
"""Checks the engine's chat prompts (TASK-30, TASK-38) against transformers:
renders a set of conversations both ways and compares the prompt text, and,
when DIR has a tokenizer.json, the prompt token ids.

transformers renders with its own Jinja environment (`render_jinja_template`,
what `apply_chat_template` calls) and the special tokens from DIR's
tokenizer_config.json; ids come from Hugging Face `tokenizers` with no special
tokens added, as `apply_chat_template` tokenizes.

Usage:
  python3 tools/ref/check_chat.py --hf-dir DIR --tftok zig-out/bin/tftok \
      --source MODEL.gguf|tokenizer.json [--config tokenizer_config.json]
  python3 tools/ref/check_chat.py --hf-dir DIR --tftok zig-out/bin/tftok --template-only

The template is DIR's chat_template.jinja, else its tokenizer_config.json's.
With --gguf-template MODEL.gguf, transformers renders with that file's template
instead, which may be an older revision than DIR's. With --template-only the
engine renders DIR's template file without a tokenizer (models it cannot load).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import render_jinja_template


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
    # TASK-38: gpt-oss's reasoning field, and a reply that ends the conversation.
    [u("Q"), {"role": "assistant", "content": "A", "thinking": "t"}],
    [s("sys"), u("Q1"), {"role": "assistant", "content": "A1", "thinking": "t1"}, u("Q2")],
]

SETTINGS = [
    {"add_generation_prompt": True},
    {"add_generation_prompt": False},
    {"add_generation_prompt": True, "enable_thinking": False},
    {"add_generation_prompt": True, "enable_thinking": True},
    {"add_generation_prompt": True, "reasoning_effort": "low"},
    {"add_generation_prompt": True, "reasoning_effort": "medium"},
    {"add_generation_prompt": True, "reasoning_effort": "high"},
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


def hf_files(d):
    """The template and special tokens transformers would use from DIR."""
    cfg = {}
    if os.path.exists(os.path.join(d, "tokenizer_config.json")):
        cfg = json.load(open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8"))
    specials = {}
    for name in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]:
        v = cfg.get(name)
        if isinstance(v, dict):
            v = v.get("content")
        if isinstance(v, str):
            specials[name] = v
    stm = os.path.join(d, "special_tokens_map.json")
    if os.path.exists(stm):
        for name, v in json.load(open(stm, encoding="utf-8")).items():
            if isinstance(v, dict):
                v = v.get("content")
            if isinstance(v, str) and name != "additional_special_tokens":
                specials.setdefault(name, v)
    jinja_path = os.path.join(d, "chat_template.jinja")
    template = open(jinja_path, encoding="utf-8").read() if os.path.exists(jinja_path) else cfg.get("chat_template")
    if isinstance(template, list):
        template = {t["name"]: t["template"] for t in template}["default"]
    return template, specials, jinja_path


def hf_render(template, specials, conv, st):
    out = render_jinja_template(conversations=[conv], chat_template=template, **specials, **st)
    # transformers returns the rendered list, with generation indices in some versions.
    if isinstance(out, tuple):
        out = out[0]
    return out[0]


def tftok_lines(cmd, files, flag):
    args = list(cmd)
    for f in files:
        args += [flag, f]
    run = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        sys.exit(1)
    if run.stderr:
        print(run.stderr.strip(), file=sys.stderr)
    return run.stdout.split("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--tftok", required=True)
    ap.add_argument("--source")
    ap.add_argument("--config")
    ap.add_argument("--gguf-template", help="render with this GGUF's template instead of DIR's")
    ap.add_argument("--template-only", action="store_true", help="the engine renders DIR's template file without a tokenizer")
    args = ap.parse_args()

    template, specials, jinja_path = hf_files(args.hf_dir)
    if args.gguf_template:
        template = gguf_template(args.gguf_template)
    if not template:
        raise SystemExit(f"{args.hf_dir} has no chat template")
    tok_path = os.path.join(args.hf_dir, "tokenizer.json")
    hf_tok = Tokenizer.from_file(tok_path) if os.path.exists(tok_path) and not args.template_only else None

    if args.template_only:
        path = jinja_path
        if not os.path.exists(path):
            path = os.path.join(tempfile.mkdtemp(), "template.jinja")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(template)
        cmd = [os.path.abspath(args.tftok), "--template", path, "--config", os.path.join(args.hf_dir, "tokenizer_config.json")]
    else:
        cmd = [os.path.abspath(args.tftok), args.source]
        if args.config:
            cmd += ["--config", args.config]

    cases = [(c, st) for c in CONVERSATIONS for st in SETTINGS]
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, (conv, st) in enumerate(cases):
            p = os.path.join(d, f"{i}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"messages": conv, **st}, f)
            files.append(p)
        texts = tftok_lines(cmd, files, "--render")
        ids = tftok_lines(cmd, files, "--chat") if hf_tok else None

    bad = 0
    refused = 0
    id_bad = 0
    for k, ((conv, st), line) in enumerate(zip(cases, texts)):
        try:
            want = hf_render(template, specials, conv, st)
        except Exception as e:
            # Templates may refuse a conversation (raise_exception); so must the engine.
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
        got = json.loads(line)
        if got != want:
            bad += 1
            if bad <= 5:
                at = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
                print(f"MISMATCH {st} {conv}")
                print(f"  first difference at character {at}")
                print(f"  transformers: {want[max(0, at - 60):at + 60]!r}")
                print(f"  engine:       {got[max(0, at - 60):at + 60]!r}")
            continue
        if ids is not None:
            want_ids = hf_tok.encode(want, add_special_tokens=False).ids
            got_ids = [int(x) for x in ids[k].split()]
            if got_ids != want_ids:
                id_bad += 1
                if id_bad <= 3:
                    print(f"ID MISMATCH {st} {conv}: {len(got_ids)} engine ids, {len(want_ids)} from tokenizers")
    summary = f"{len(cases) - bad}/{len(cases)} prompts identical ({refused} refused by both)"
    if ids is not None:
        summary += f"; token ids identical for {len(cases) - bad - refused - id_bad} of {len(cases) - bad - refused}"
    print(summary)
    sys.exit(1 if bad or id_bad else 0)


if __name__ == "__main__":
    main()
