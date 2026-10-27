#!/usr/bin/env python3
"""TASK-06 reference harness: a tiny synthetic MoE in PyTorch (float32, CPU)
that writes weights, inputs, and golden intermediate and final outputs to a
.tfref file. Testing only: the engine never links Python or PyTorch.

Model: L pre-norm MoE blocks with no attention, then a final norm and an LM
head (OLMoE-style routing: softmax over all experts, top-k, no
renormalization; SwiGLU experts):

    h = x
    for each layer:
        n      = rmsnorm(h) * norm_w
        p      = softmax(n @ router_w)            # [T, E]
        w, idx = topk(p, k)                       # sorted descending
        h      = h + sum_j w[:, j] * expert[idx[:, j]](n)
    logits = (rmsnorm(h) * final_norm_w) @ lm_head

    expert(n) = (silu(n @ gate) * (n @ up)) @ down

Usage (from the repo root):
  python3 tools/ref/ref_moe.py                 # write tests/golden/moe_tiny.tfref
  python3 tools/ref/ref_moe.py --check         # regenerate and compare with it
"""
import argparse
import hashlib
import os
import struct
import sys

import torch

CONFIG = dict(layers=2, experts=8, top_k=2, d_model=32, d_ff=64, vocab=128, tokens=16, seed=20261101, eps=1e-5)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "..", "tests", "golden", "moe_tiny.tfref")

# .tfref container, all little-endian:
#   header  : magic "TFREF\0\0\0", u32 version, u32 count
#   entries : u16 name_len, name (UTF-8), u8 dtype (0 f32, 1 i32), u8 ndim,
#             u32 dims[ndim], u64 offset, u64 nbytes
#   data    : each tensor at `offset` from file start, 64-byte aligned
MAGIC = b"TFREF\0\0\0"
VERSION = 1
DTYPES = {torch.float32: 0, torch.int32: 1}


def build(cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    d, f, e = cfg["d_model"], cfg["d_ff"], cfg["experts"]

    def rand(*shape, scale):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * scale

    t = {}
    for l in range(cfg["layers"]):
        t[f"layer{l}.norm_w"] = 1.0 + rand(d, scale=0.1)
        t[f"layer{l}.router_w"] = rand(d, e, scale=d ** -0.5)
        t[f"layer{l}.gate"] = rand(e, d, f, scale=d ** -0.5)
        t[f"layer{l}.up"] = rand(e, d, f, scale=d ** -0.5)
        t[f"layer{l}.down"] = rand(e, f, d, scale=f ** -0.5)
    t["final_norm_w"] = 1.0 + rand(d, scale=0.1)
    t["lm_head"] = rand(d, cfg["vocab"], scale=d ** -0.5)
    t["input"] = rand(cfg["tokens"], d, scale=1.0)
    return t


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def forward(cfg, t):
    out = {}
    h = t["input"]
    for l in range(cfg["layers"]):
        n = rmsnorm(h, t[f"layer{l}.norm_w"], cfg["eps"])
        logits = n @ t[f"layer{l}.router_w"]
        probs = torch.softmax(logits, dim=-1)
        w, idx = torch.topk(probs, cfg["top_k"], dim=-1, sorted=True)
        moe = torch.zeros_like(h)
        for tok in range(h.shape[0]):
            for j in range(cfg["top_k"]):
                ex = int(idx[tok, j])
                a = n[tok] @ t[f"layer{l}.gate"][ex]
                b = n[tok] @ t[f"layer{l}.up"][ex]
                moe[tok] += w[tok, j] * ((torch.nn.functional.silu(a) * b) @ t[f"layer{l}.down"][ex])
        h = h + moe
        out[f"golden.layer{l}.router_logits"] = logits
        out[f"golden.layer{l}.topk_idx"] = idx.to(torch.int32)
        out[f"golden.layer{l}.topk_w"] = w
        out[f"golden.layer{l}.moe_out"] = moe
        out[f"golden.layer{l}.hidden"] = h
    out["golden.logits"] = rmsnorm(h, t["final_norm_w"], cfg["eps"]) @ t["lm_head"]
    return out


def config_tensor(cfg):
    keys = ["layers", "experts", "top_k", "d_model", "d_ff", "vocab", "tokens", "seed"]
    return torch.tensor([cfg[k] for k in keys], dtype=torch.int32)


def serialize(tensors):
    names = list(tensors)
    header = bytearray(MAGIC + struct.pack("<II", VERSION, len(names)))
    # Entry sizes are known up front, so offsets can be assigned in one pass.
    entry_bytes = sum(2 + len(n.encode()) + 2 + 4 * tensors[n].dim() + 16 for n in names)
    offset = len(header) + entry_bytes
    blobs = []
    for n in names:
        x = tensors[n].contiguous()
        offset = (offset + 63) // 64 * 64
        data = x.numpy().tobytes()  # host order; main() requires a little-endian host
        header += struct.pack("<H", len(n.encode())) + n.encode()
        header += struct.pack("<BB", DTYPES[x.dtype], x.dim()) + struct.pack("<%dI" % x.dim(), *x.shape)
        header += struct.pack("<QQ", offset, len(data))
        blobs.append((offset, data))
        offset += len(data)
    out = bytearray(header)
    for off, data in blobs:
        out += b"\0" * (off - len(out)) + data
    return bytes(out)


def parse(blob):
    assert blob[:8] == MAGIC, "not a .tfref file"
    version, count = struct.unpack_from("<II", blob, 8)
    assert version == VERSION
    pos, out = 16, {}
    for _ in range(count):
        (nlen,) = struct.unpack_from("<H", blob, pos)
        name = blob[pos + 2:pos + 2 + nlen].decode()
        pos += 2 + nlen
        dtype, ndim = struct.unpack_from("<BB", blob, pos)
        dims = struct.unpack_from("<%dI" % ndim, blob, pos + 2)
        off, nbytes = struct.unpack_from("<QQ", blob, pos + 2 + 4 * ndim)
        pos += 2 + 4 * ndim + 16
        tdtype = torch.float32 if dtype == 0 else torch.int32
        out[name] = torch.frombuffer(bytearray(blob[off:off + nbytes]), dtype=tdtype).reshape(dims)
    return out


def generate(cfg):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    t = build(cfg)
    tensors = {"config": config_tensor(cfg)}
    tensors.update(t)
    tensors.update(forward(cfg, t))
    return serialize(tensors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="regenerate and compare with --out instead of writing")
    args = ap.parse_args()
    if sys.byteorder != "little":
        sys.exit(".tfref is little-endian; run on a little-endian host")

    blob = generate(CONFIG)
    if not args.check:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "wb") as f:
            f.write(blob)
        print("wrote %s (%d bytes, sha256 %s, torch %s)" % (
            os.path.abspath(args.out), len(blob), hashlib.sha256(blob).hexdigest()[:16], torch.__version__))
        return

    with open(args.out, "rb") as f:
        old = f.read()
    if old == blob:
        print("check: byte-identical to %s" % args.out)
        return
    # Weights and inputs come straight from the seeded generator and must match
    # exactly; golden outputs may differ in the last bits across BLAS builds.
    a, b = parse(old), parse(blob)
    worst = 0.0
    for name in a:
        if not name.startswith("golden.") or a[name].dtype == torch.int32:
            if not torch.equal(a[name], b[name]):
                sys.exit("check: %s differs" % name)
        else:
            rel = ((a[name] - b[name]).abs() / a[name].abs().clamp_min(1e-6)).max().item()
            worst = max(worst, rel)
    if worst > 1e-5:
        sys.exit("check: golden outputs differ, max relative error %.3g" % worst)
    print("check: weights and inputs identical, outputs within %.3g relative" % worst)


if __name__ == "__main__":
    main()
