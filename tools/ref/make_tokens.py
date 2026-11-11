#!/usr/bin/env python3
"""Tokenizes text files into one long prompt for the long-context runs
(TASK-12): writes the token ids as little-endian u32, the format
`olmoe --tokens` reads. Testing only.

Usage:
  python3 tools/ref/make_tokens.py --tokenizer tokenizer.json --out long.tokens FILE...
"""
import argparse
import struct

from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    ids = []
    for path in args.files:
        with open(path, encoding="utf-8", errors="replace") as f:
            ids.extend(tok.encode(f.read() + "\n\n").ids)
    with open(args.out, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}I", *ids))
    print(f"wrote {args.out}: {len(ids)} tokens from {len(args.files)} files")


if __name__ == "__main__":
    main()
