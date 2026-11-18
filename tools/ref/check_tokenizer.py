#!/usr/bin/env python3
"""Checks the engine's tokenizer (TASK-30) against Hugging Face `tokenizers`:
the golden prompts, hand-picked edge cases, seeded random strings, and any
long text files given. Runs `tftok` on the same texts and compares ids.

Usage:
  python3 tools/ref/check_tokenizer.py --tokenizer tokenizer.json \
      --tftok zig-out/bin/tftok --source MODEL.gguf|tokenizer.json [--long FILE]...
"""
import argparse
import ast
import os
import random
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

EDGE = [
    "",
    " ",
    "\n",
    "a  b",
    "x" + " " * 30 + "y",
    "def f():\n        return 1\n\n\n\tpass\r\n",
    "It's it'S IT'RE we've I'm you'll he'd 'ſ ''s",
    "café é Å Å ḍ̇",
    "각 한국어",
    "नमस्ते दुनिया مَرْحَبًا ‿x ̈y",
    "中文字符 日本語のテキスト",
    "emoji \U0001F600\U0001F44D\U0001F3FD family \U0001F468‍\U0001F469‍\U0001F467",
    "numbers 12345 ١٢٣ Ⅳ ½ 3.14159 1,000,000",
    "spaces  　and lines end",
    "<|endoftext|>hi<|im_start|>user\nhello<|im_end|>\n<think>\n</think>",
    "|||IP_ADDRESS||| and |||EMAIL_ADDRESS|||",
    "trailing spaces   ",
    "   leading spaces",
    "\n\n\n  \n\t\n",
    "!!!???...,,,;;;",
    "mixed\r\nline\rendings\n",
    "̀́ starts with combining marks",
    "Привет, мир! Γειά σου Κόσμε! שלום עולם مرحبا بالعالم",
    # TASK-37: case splits (gpt-oss), CJK runs and ASCII punctuation (DeepSeek V3), SentencePiece spaces and specials.
    "HelloWORLDfoo camelCaseWord XMLHttpRequest ǅungla ʰello DON'T can'T I'M",
    "path/to/file\n/usr/bin\r\n// comment\n",
    "a1b22c333d4444 x١٢٣ 3.14 1e-10",
    "日本語のテキストabc漢字カタカナ㐀𠀀 中文，标点。",
    "#include <stdio.h> e.g. U.S.A. $100 @user #tag ~home",
    "!!abc ??Def ...xyz ,a ;b",
    "▁ hi ▁▁there",
    "<s>hello</s> world<unk> <s> spaced",
    "\x00\x01 control\x7f chars​ zero width",
]

POOL = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + list(" \t\n\r'\".,;:!?()[]{}<>-_=+*/\\|@#$%^&~`")
    + ["  ", "    ", " ", "　", " ", "\u0085"]
    + ["é", "é", "́", "̣", "Å", "Å", "ſ", "ß"]
    + ["中", "文", "あ", "한", "ᄀ", "ᅡ", "ᆨ"]
    + ["١", "Ⅳ", "½", "\U0001F600", "‍", "️", "\U0001F3FD"]
    + ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "'s", "'LL"]
)


def golden_prompts():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_olmoe.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("PROMPTS not found in ref_olmoe.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer.json")
    ap.add_argument("--tftok", required=True)
    ap.add_argument("--source", required=True, help="what tftok loads: GGUF or tokenizer.json")
    ap.add_argument("--long", action="append", default=[])
    ap.add_argument("--random", type=int, default=500)
    args = ap.parse_args()

    rng = random.Random(30)
    texts = [("golden", p) for p in golden_prompts()]
    texts += [("edge", t) for t in EDGE]
    for _ in range(args.random):
        texts.append(("random", "".join(rng.choice(POOL) for _ in range(rng.randint(1, 40)))))
    for path in args.long:
        with open(path, encoding="utf-8") as f:
            texts.append((os.path.basename(path), f.read()))

    hf = Tokenizer.from_file(args.tokenizer)
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, (_, t) in enumerate(texts):
            p = os.path.join(d, f"{i}.txt")
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(t)
            files.append(p)
        run = subprocess.run([os.path.abspath(args.tftok), args.source] + files, capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
    lines = run.stdout.split("\n")

    bad = 0
    total = 0
    for (kind, text), line in zip(texts, lines):
        got = [int(x) for x in line.split()]
        want = hf.encode(text).ids
        total += len(want)
        if got != want:
            bad += 1
            if bad <= 10:
                at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                print(f"MISMATCH ({kind}) {text[:80]!r}")
                print(f"  at token {at}: engine {got[max(0, at - 3):at + 5]}")
                print(f"  {'':>{len(str(at)) + 10}}hf     {want[max(0, at - 3):at + 5]}")
                print(f"  hf tokens {[hf.id_to_token(i) for i in want[max(0, at - 3):at + 5]]}")
    print(f"{len(texts) - bad}/{len(texts)} texts identical ({total} tokens); decode round trip {'ok' if run.returncode == 0 else 'FAILED'}")
    sys.exit(1 if bad or run.returncode != 0 else 0)


if __name__ == "__main__":
    main()
