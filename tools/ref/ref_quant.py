#!/usr/bin/env python3
"""TASK-08 reference harness for the expert loader. Testing only. Writes two
tiny MoE-shaped checkpoints and, for each, a .tfref file of the values the
engine's loader must get from them:

  quant_tiny.gguf          llama.cpp layout, stacked expert tensors in Q8_0,
                           MXFP4, and NVFP4 (plus F32 and F16 tensors).
                           Written with gguf-py's GGUFWriter and quantizers;
                           golden values from gguf-py's own dequantizer.
  nvfp4_tiny.safetensors   NVIDIA ModelOpt NVFP4 export layout (per-expert
                           packed E2M1 weight, E4M3 weight_scale per 16
                           values, f32 weight_scale_2), plus config.json.
                           Golden values decode the E4M3 scales with
                           PyTorch's float8_e4m3fn.

  --model FILE.gguf --out OUT.tfref [--experts 0,17,63] [--layers 0,15]
      writes golden values for those experts of a real GGUF model, from
      gguf-py's dequantizer, for `tfload load --golden`.

Usage (from the repo root):
  python3 tools/ref/ref_quant.py           # write the fixtures to tests/golden/
  python3 tools/ref/ref_quant.py --check   # regenerate and compare
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q
import safetensors.torch
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402  (the shared .tfref writer)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "..", "tests", "golden")
SEED = 20261105
EXPERTS, D_MODEL, D_FF = 4, 64, 128
PROJ = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
FILES = ("quant_tiny.gguf", "quant_tiny.tfref", "nvfp4_tiny.safetensors", "nvfp4_tiny.config.json", "nvfp4_tiny.tfref")

# GGUF: name -> (numpy shape, outermost first, format). Stacked expert tensors
# are [experts, rows, row_len], as in llama.cpp's OLMoE conversion.
GGUF_TENSORS = {
    "blk.0.ffn_gate_exps.weight": ((EXPERTS, D_FF, D_MODEL), Q.Q8_0),
    "blk.0.ffn_up_exps.weight": ((EXPERTS, D_FF, D_MODEL), Q.MXFP4),
    "blk.0.ffn_down_exps.weight": ((EXPERTS, D_MODEL, D_FF), Q.NVFP4),
    "blk.0.ffn_gate_inp.weight": ((EXPERTS, D_MODEL), Q.F32),
    "blk.0.attn_norm.weight": ((D_MODEL,), Q.F16),
}


def quantize_nvfp4_gguf(x):
    """gguf-py decodes NVFP4 but cannot encode it. Blocks of 64 values: four
    unsigned E4M3 scales (one per 16 values, encoded with gguf-py's own
    fp32_to_ue4m3), then per 16 values 8 bytes, low nibbles first. The golden
    values come from gguf-py's decoder, not from this function."""
    groups = x.reshape(-1, 16)
    scale = gguf.quants.NVFP4.fp32_to_ue4m3(np.abs(groups).max(axis=1) / 6.0)
    d = gguf.quants.NVFP4.ue4m3_to_fp32(scale).reshape(-1, 1)  # halved, like the doubled codes
    codes = np.array(gguf.quants.NVFP4.kvalues, dtype=np.float32)
    best = np.abs(d[:, :, None] * codes[None, None, :] - groups[:, :, None]).argmin(axis=-1).astype(np.uint8)
    packed = best[:, :8] | (best[:, 8:] << np.uint8(4))  # [groups, 8]
    blocks = np.concatenate([scale.reshape(-1, 4), packed.reshape(-1, 32)], axis=1)
    return blocks.reshape(*x.shape[:-1], x.shape[-1] // 64 * 36)


def write_gguf(out_dir):
    rng = np.random.default_rng(SEED)
    w = gguf.GGUFWriter(path=None, arch="olmoe")
    w.add_block_count(1)
    w.add_embedding_length(D_MODEL)
    w.add_expert_count(EXPERTS)
    w.add_expert_used_count(2)
    w.add_expert_feed_forward_length(D_FF)
    golden = {}
    for name, (shape, qtype) in GGUF_TENSORS.items():
        x = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if qtype == Q.F32:
            w.add_tensor(name, x)
            y = x
        elif qtype == Q.F16:
            h = x.astype(np.float16)
            w.add_tensor(name, h)
            y = h.astype(np.float32)
        else:
            q = quantize_nvfp4_gguf(x) if qtype == Q.NVFP4 else gguf.quants.quantize(x, qtype)
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=qtype)
            y = gguf.quants.dequantize(q, qtype).reshape(shape).astype(np.float32)
        golden[f"golden.{name}"] = y.reshape(-1)
        for p, stem in PROJ.items():
            if name == f"blk.0.{stem}.weight":
                for e in range(EXPERTS):
                    golden[f"golden.layer0.{p}.expert{e}"] = y[e].reshape(-1)
    path = os.path.join(out_dir, "quant_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    write_tfref(os.path.join(out_dir, "quant_tiny.tfref"), golden)


# ModelOpt NVFP4. E2M1 magnitudes for codes 0..7; bit 3 is the sign.
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def quantize_modelopt_nvfp4(x):
    """x: [rows, cols] float32. Returns (packed u8 [rows, cols/2], scale
    float8_e4m3fn [rows, cols/16], global f32 scalar), following ModelOpt:
    global = amax / (6 * 448); block scale = block amax / 6 / global."""
    rows, cols = x.shape
    glob = (x.abs().max() / (6.0 * 448.0)).to(torch.float32)
    blocks = x.reshape(rows, cols // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / 6.0 / glob).to(torch.float8_e4m3fn)
    d = scale.to(torch.float32).unsqueeze(-1) * glob
    mag = (blocks.abs() / torch.where(d == 0, torch.ones_like(d), d)).unsqueeze(-1)
    code = (mag - E2M1).abs().argmin(dim=-1).to(torch.uint8) | ((blocks < 0).to(torch.uint8) << 3)
    code = code.reshape(rows, cols)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)  # low nibble first
    return packed.contiguous(), scale.contiguous(), glob.reshape(())


def dequantize_modelopt_nvfp4(packed, scale, glob):
    lo, hi = packed & 0xF, packed >> 4
    code = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1).long()
    val = E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    rows, cols = val.shape
    d = scale.to(torch.float32) * glob  # PyTorch's own E4M3 decode
    return (val.reshape(rows, cols // 16, 16) * d.unsqueeze(-1)).reshape(rows, cols)


def write_safetensors(out_dir):
    g = torch.Generator().manual_seed(SEED + 1)
    layers = 2
    config = {"architectures": ["OlmoeForCausalLM"], "num_hidden_layers": layers, "num_experts": EXPERTS,
              "hidden_size": D_MODEL, "intermediate_size": D_FF,
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16}}
    tensors, golden = {}, {}
    for l in range(layers):
        for e in range(EXPERTS):
            for p in ("gate", "up", "down"):
                shape = (D_MODEL, D_FF) if p == "down" else (D_FF, D_MODEL)
                x = torch.randn(*shape, generator=g) * 0.05
                packed, scale, glob = quantize_modelopt_nvfp4(x)
                base = f"model.layers.{l}.mlp.experts.{e}.{p}_proj"
                tensors[f"{base}.weight"] = packed
                tensors[f"{base}.weight_scale"] = scale
                tensors[f"{base}.weight_scale_2"] = glob
                tensors[f"{base}.input_scale"] = torch.tensor(1.0)  # present in real exports; unused
                golden[f"golden.layer{l}.{p}.expert{e}"] = dequantize_modelopt_nvfp4(packed, scale, glob).reshape(-1).numpy()
    safetensors.torch.save_file(tensors, os.path.join(out_dir, "nvfp4_tiny.safetensors"))
    with open(os.path.join(out_dir, "nvfp4_tiny.config.json"), "w", newline="\n") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    write_tfref(os.path.join(out_dir, "nvfp4_tiny.tfref"), golden)


def write_tfref(path, arrays):
    with open(path, "wb") as f:
        f.write(serialize({k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in arrays.items()}))


def write_model_golden(model, out, layers, experts):
    r = gguf.GGUFReader(model)
    by_name = {t.name: t for t in r.tensors}
    golden = {}
    for l in layers:
        for p, stem in PROJ.items():
            t = by_name[f"blk.{l}.{stem}.weight"]
            for e in experts:
                # t.data is [experts, rows, row bytes] for quantized types.
                y = gguf.quants.dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float32)
                golden[f"golden.layer{l}.{p}.expert{e}"] = y.reshape(-1)
    write_tfref(out, golden)
    print(f"wrote {out}: {len(golden)} expert weights from {os.path.basename(model)}")


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--experts", default="0")
    args = ap.parse_args()
    if sys.byteorder != "little":
        sys.exit("run on a little-endian host")
    if args.model:
        ints = lambda s: [int(v) for v in s.split(",")]  # noqa: E731
        write_model_golden(args.model, args.out, ints(args.layers), ints(args.experts))
        return
    if not args.check:
        write_gguf(GOLDEN)
        write_safetensors(GOLDEN)
        for n in FILES:
            p = os.path.join(GOLDEN, n)
            print(f"wrote {os.path.relpath(p)} ({os.path.getsize(p)} bytes)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        write_gguf(tmp)
        write_safetensors(tmp)
        bad = [n for n in FILES if digest(os.path.join(tmp, n)) != digest(os.path.join(GOLDEN, n))]
    if bad:
        sys.exit(f"MISMATCH: {', '.join(bad)} differ from a fresh run")
    print("OK: every fixture matches a fresh run")


if __name__ == "__main__":
    main()
