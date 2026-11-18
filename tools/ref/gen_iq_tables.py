"""Generates the IQ lookup tables (TASK-36) from llama.cpp's ggml-common.h.

Usage: python gen_iq_tables.py path/to/ggml-common.h
Writes src/iq_tables.zig and csrc/cuda/iq_tables.cuh, each carrying ggml's MIT notice.
"""
import pathlib
import re
import sys

TABLES = [
    ("ksigns_iq2xs", "u8", "unsigned char", 128),
    ("iq2xxs_grid", "u64", "unsigned long long", 256),
    ("iq2xs_grid", "u64", "unsigned long long", 512),
    ("iq2s_grid", "u64", "unsigned long long", 1024),
    ("iq3xxs_grid", "u32", "unsigned int", 256),
    ("iq3s_grid", "u32", "unsigned int", 512),
    ("iq1s_grid", "u64", "unsigned long long", 2048),
    ("kvalues_iq4nl", "i8", "signed char", 16),
]

NOTICE = """Copied from ggml (llama.cpp, ggml/src/ggml-common.h) by tools/ref/gen_iq_tables.py; do not edit.
MIT License. Copyright (c) 2023-2026 The ggml authors.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""


def table(src, name, size):
    m = re.search(r"GGML_TABLE_BEGIN\(\w+, " + name + r", \w+\)(.*?)GGML_TABLE_END\(\)", src, re.S)
    if not m:
        sys.exit(f"{name} not found")
    vals = [int(v, 0) for v in re.findall(r"-?(?:0x[0-9a-fA-F]+|\d+)", m.group(1))]
    if len(vals) != size:
        sys.exit(f"{name}: {len(vals)} values, expected {size}")
    return vals


def fmt(v, kind):
    return f"0x{v:016x}" if kind == "u64" else f"0x{v:08x}" if kind == "u32" else str(v)


def rows(vals, kind, per):
    out = []
    for i in range(0, len(vals), per):
        out.append("    " + ", ".join(fmt(v, kind) for v in vals[i:i + per]) + ",")
    return out


def main():
    src = pathlib.Path(sys.argv[1]).read_text()
    root = pathlib.Path(__file__).resolve().parents[2]
    zig = ["//! " + line for line in NOTICE.splitlines()]
    cu = ["// " + line for line in NOTICE.splitlines()] + ["#pragma once"]
    for name, kind, ctype, size in TABLES:
        vals = table(src, name, size)
        per = 4 if kind == "u64" else 8 if kind == "u32" else 16
        zig += ["", f"pub const {name} = [{size}]{kind}{{"] + rows(vals, kind, per) + ["};"]
        cu += ["", f"static __device__ const {ctype} {name}[{size}] = {{"] + rows(vals, kind, per) + ["};"]
    (root / "src" / "iq_tables.zig").write_text("\n".join(zig) + "\n", newline="\n")
    (root / "csrc" / "cuda" / "iq_tables.cuh").write_text("\n".join(cu) + "\n", newline="\n")


main()
