#!/usr/bin/env python3
"""TASK-29 reference for Qwen3.6-35B-A3B (transformers' qwen3_5_moe): runs
Hugging Face's own Qwen3_5MoeForCausalLM (float32, eager attention, CPU) on
the weights in a llama.cpp GGUF file and writes the same golden values as
ref_olmoe.py: per prompt, the token ids, every position's logits, each
layer's output hidden state and top-k routing, and optionally greedy
generate(). Testing only.

The GGUF holds the weights as llama.cpp's converter left them, so loading
undoes its changes: RMSNorm gammas were stored plus one (all but the gated
norm), A_log as -exp(A_log), dt_bias renamed, conv1d squeezed, and the value
heads of the linear-attention layers moved from grouped order (each key
head's value heads together) to tiled order (value head t uses key head
t mod key heads). As in ref_olmoe.py, the experts stay quantized and
dequantize from the memory-mapped file when called.

Usage:
  python3 tools/ref/ref_qwen35.py --gguf Qwen3.6-35B-A3B-Q8_0.gguf --config config.json \\
      --tokenizer tokenizer.json --out qwen35_golden.tfref [--generate 32]
  python3 tools/ref/ref_qwen35.py --tiny [--check]
      writes tests/golden/qwen35_tiny.{gguf,tfref}: a tiny model with the
      same layer mix and random weights (seeded), and transformers' outputs.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import gguf
from tokenizers import Tokenizer
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS, record  # noqa: E402


def tiled_to_grouped(x, dim, k_heads, v_per_k, head_dim):
    """Undoes llama.cpp's value-head reorder along `dim` (a [v_heads * head_dim] axis)."""
    shape = list(x.shape)
    x = x.reshape(shape[:dim] + [v_per_k, k_heads, head_dim] + shape[dim + 1:])
    return x.transpose(dim, dim + 1).contiguous().reshape(shape)


def grouped_to_tiled(x, dim, k_heads, v_per_k, head_dim):
    shape = list(x.shape)
    x = x.reshape(shape[:dim] + [k_heads, v_per_k, head_dim] + shape[dim + 1:])
    return x.transpose(dim, dim + 1).contiguous().reshape(shape)


class LazyExperts(torch.nn.Module):
    """A layer's experts, each dequantized from the stacked GGUF tensors when a token picks it."""

    def __init__(self, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down

    @staticmethod
    def weight(t, e):
        if callable(t):
            return t(e)
        w = gguf.quants.dequantize(np.asarray(t.data[e]), t.tensor_type)
        return torch.from_numpy(w.astype(np.float32))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            tok, pos = torch.where(top_k_index == e)
            x = hidden_states[tok]
            h = torch.nn.functional.silu(x @ self.weight(self.gate, e).T) * (x @ self.weight(self.up, e).T)
            y = (h @ self.weight(self.down, e).T) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


def hf_state(cfg, tensors):
    """transformers' state dict (without experts) from GGUF tensors given as name -> float32 numpy (outermost first)."""
    k_heads, v_heads = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    r, hv = v_heads // k_heads, cfg.linear_value_head_dim
    key_dim = cfg.linear_key_head_dim * k_heads

    def t(name):
        return torch.from_numpy(np.ascontiguousarray(tensors[name]))

    def norm(name):
        return t(name) - 1.0

    def untile(x, dim, head_dim):
        return tiled_to_grouped(x, dim, k_heads, r, head_dim)

    sd = {"model.embed_tokens.weight": t("token_embd.weight"), "model.norm.weight": norm("output_norm.weight"),
          "lm_head.weight": t("output.weight")}
    for l, kind in enumerate(cfg.layer_types):
        g, h = f"blk.{l}.", f"model.layers.{l}."
        sd[h + "input_layernorm.weight"] = norm(g + "attn_norm.weight")
        sd[h + "post_attention_layernorm.weight"] = norm(g + "post_attention_norm.weight")
        if kind == "linear_attention":
            a = h + "linear_attn."
            qkv = t(g + "attn_qkv.weight")
            qkv = torch.cat([qkv[:2 * key_dim], untile(qkv[2 * key_dim:], 0, hv)])
            sd[a + "in_proj_qkv.weight"] = qkv
            sd[a + "in_proj_z.weight"] = untile(t(g + "attn_gate.weight"), 0, hv)
            sd[a + "in_proj_b.weight"] = untile(t(g + "ssm_beta.weight"), 0, 1)
            sd[a + "in_proj_a.weight"] = untile(t(g + "ssm_alpha.weight"), 0, 1)
            sd[a + "A_log"] = torch.log(-untile(t(g + "ssm_a"), 0, 1))
            sd[a + "dt_bias"] = untile(t(g + "ssm_dt.bias"), 0, 1)
            conv = t(g + "ssm_conv1d.weight")
            conv = torch.cat([conv[:2 * key_dim], untile(conv[2 * key_dim:], 0, hv)])
            sd[a + "conv1d.weight"] = conv.unsqueeze(1)
            sd[a + "norm.weight"] = t(g + "ssm_norm.weight")
            sd[a + "out_proj.weight"] = untile(t(g + "ssm_out.weight"), 1, hv)
        else:
            a = h + "self_attn."
            for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                sd[a + hf + ".weight"] = t(g + gg + ".weight")
            sd[a + "q_norm.weight"] = norm(g + "attn_q_norm.weight")
            sd[a + "k_norm.weight"] = norm(g + "attn_k_norm.weight")
        m = h + "mlp."
        sd[m + "gate.weight"] = t(g + "ffn_gate_inp.weight")
        for hf, gg in (("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")):
            sd[m + "shared_expert." + hf + ".weight"] = t(g + gg + ".weight")
        sd[m + "shared_expert_gate.weight"] = t(g + "ffn_gate_inp_shexp.weight").reshape(1, -1)
    return sd


def load(cfg, tensors, experts):
    """The model on these weights; `experts(layer)` gives (gate, up, down) for LazyExperts."""
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(cfg)
    missing = model.load_state_dict(hf_state(cfg, tensors), strict=False, assign=True)
    for l, layer in enumerate(model.model.layers):
        layer.mlp.experts = LazyExperts(*experts(l))
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    # Rotary inverse frequencies are buffers built on the meta device; rebuild them.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    return model.eval()


def dequant(t):
    shape = tuple(int(d) for d in reversed(t.shape))
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(t.data, dtype=np.float32).reshape(shape)
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(shape).astype(np.float32)


def text_config(path):
    with open(path) as f:
        c = json.load(f)
    cfg = Qwen3_5MoeTextConfig(**c.get("text_config", c))
    cfg.dtype = torch.float32
    return cfg


def build(gguf_path, config_path):
    cfg = text_config(config_path)
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    tensors = {t.name: dequant(t) for t in reader.tensors if "_exps." not in t.name}
    experts = lambda l: tuple(by_name[f"blk.{l}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))  # noqa: E731
    return load(cfg, tensors, experts), cfg


TINY = dict(vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
            linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2,
            moe_intermediate_size=32, shared_expert_intermediate_size=32, rms_norm_eps=1e-6,
            max_position_embeddings=64, full_attention_interval=4, tie_word_embeddings=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_interleaved": True, "mrope_section": [2, 1, 1]})
TINY_SEED = 20261116
TINY_PROMPTS = [[3, 17, 200, 45, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [250, 128, 64, 32, 16, 8, 4]]
TINY_GENERATE = 16
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir):
    """A tiny Qwen3.5-MoE (three linear-attention layers, then one gated full-attention
    layer) with random weights, written in llama.cpp's layout: every 2-D weight in
    Q8_0, and the reference runs on the dequantized values through `load`."""
    cfg = Qwen3_5MoeTextConfig(**TINY)
    cfg.dtype = torch.float32
    assert cfg.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8 = gguf.GGMLQuantizationType.Q8_0
    d, f, e, v = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.vocab_size
    hd, kh, vh = cfg.head_dim, cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    conv_dim = 2 * kh * dk + vh * dv

    w = gguf.GGUFWriter(path=None, arch="qwen35moe")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(d)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_key_length(hd)
    w.add_value_length(hd)
    w.add_expert_count(e)
    w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_expert_feed_forward_length(f)
    w.add_uint32("qwen35moe.expert_shared_feed_forward_length", cfg.shared_expert_intermediate_size)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(cfg.rope_parameters["rope_theta"])
    w.add_uint32("qwen35moe.rope.dimension_count", int(hd * 0.25))
    w.add_uint32("qwen35moe.ssm.conv_kernel", cfg.linear_conv_kernel_dim)
    w.add_uint32("qwen35moe.ssm.state_size", dk)
    w.add_uint32("qwen35moe.ssm.group_count", kh)
    w.add_uint32("qwen35moe.ssm.time_step_rank", vh)
    w.add_uint32("qwen35moe.ssm.inner_size", vh * dv)
    w.add_uint32("qwen35moe.full_attention_interval", TINY["full_attention_interval"])

    tensors = {}
    stacked = {}

    def q8(name, x):
        q = gguf.quants.quantize(x.numpy(), Q8)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q8)
        return gguf.quants.dequantize(q, Q8).reshape(x.shape).astype(np.float32)

    def linear(name, shape, scale=1.0):
        tensors[name] = q8(name, torch.randn(*shape, generator=g) * scale / shape[-1] ** 0.5)

    def f32(name, x):
        w.add_tensor(name, x.numpy().astype(np.float32))
        tensors[name] = x.numpy().astype(np.float32)

    def gamma(name, n):
        f32(name, 1.0 + torch.randn(n, generator=g) * 0.1)

    tensors["token_embd.weight"] = q8("token_embd.weight", torch.randn(v, d, generator=g))
    for l, kind in enumerate(cfg.layer_types):
        p = f"blk.{l}."
        gamma(p + "attn_norm.weight", d)
        gamma(p + "post_attention_norm.weight", d)
        if kind == "linear_attention":
            linear(p + "attn_qkv.weight", (conv_dim, d))
            linear(p + "attn_gate.weight", (vh * dv, d))
            linear(p + "ssm_beta.weight", (vh, d))
            linear(p + "ssm_alpha.weight", (vh, d))
            f32(p + "ssm_a", -torch.empty(vh).uniform_(0.5, 4, generator=g))
            f32(p + "ssm_dt.bias", torch.randn(vh, generator=g) * 0.5)
            f32(p + "ssm_conv1d.weight", torch.randn(conv_dim, cfg.linear_conv_kernel_dim, generator=g) * 0.5)
            gamma(p + "ssm_norm.weight", dv)
            linear(p + "ssm_out.weight", (d, vh * dv))
        else:
            linear(p + "attn_q.weight", (2 * cfg.num_attention_heads * hd, d))
            linear(p + "attn_k.weight", (cfg.num_key_value_heads * hd, d))
            linear(p + "attn_v.weight", (cfg.num_key_value_heads * hd, d))
            linear(p + "attn_output.weight", (d, cfg.num_attention_heads * hd))
            gamma(p + "attn_q_norm.weight", hd)
            gamma(p + "attn_k_norm.weight", hd)
        f32(p + "ffn_gate_inp.weight", torch.randn(e, d, generator=g) * (4.0 / d ** 0.5))
        f32(p + "ffn_gate_inp_shexp.weight", torch.randn(d, generator=g) / d ** 0.5)
        linear(p + "ffn_gate_shexp.weight", (cfg.shared_expert_intermediate_size, d))
        linear(p + "ffn_up_shexp.weight", (cfg.shared_expert_intermediate_size, d))
        linear(p + "ffn_down_shexp.weight", (d, cfg.shared_expert_intermediate_size))
        for gg, shape in (("gate", (f, d)), ("up", (f, d)), ("down", (d, f))):
            stacked[(l, gg)] = torch.from_numpy(q8(p + f"ffn_{gg}_exps.weight",
                                                   torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5))
    gamma("output_norm.weight", d)
    linear("output.weight", (v, d))

    experts = lambda l: tuple((lambda x, s=stacked[(l, p)]: s[x]) for p in ("gate", "up", "down"))  # noqa: E731
    model = load(cfg, tensors, experts)
    path = os.path.join(out_dir, "qwen35_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, TINY_PROMPTS, TINY_GENERATE)
    with open(os.path.join(out_dir, "qwen35_tiny.tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    names = ("qwen35_tiny.gguf", "qwen35_tiny.tfref")
    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    if not check:
        gap = write_tiny(GOLDEN_DIR)
        for n in names:
            print(f"wrote tests/golden/{n} ({os.path.getsize(os.path.join(GOLDEN_DIR, n))} bytes)")
        print(f"smallest gap between the k-th and (k+1)-th router probability: {gap:.2e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        write_tiny(tmp)
        bad = [n for n in names if digest(os.path.join(tmp, n)) != digest(os.path.join(GOLDEN_DIR, n))]
    if bad:
        sys.exit(f"MISMATCH: {', '.join(bad)} differ from a fresh run")
    print("OK: qwen35_tiny.gguf and qwen35_tiny.tfref match a fresh run")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0, help="also record this many tokens of greedy generate()")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    model, cfg = build(args.gguf, args.config)
    tok = Tokenizer.from_file(args.tokenizer)
    prompts = [tok.encode(text).ids for text in PROMPTS]
    out, gap = record(model, cfg, prompts, args.generate, gaps=True)
    for p, ids in enumerate(prompts):
        top1 = out[f"prompt{p}.logits"].argmax(-1).tolist()
        print(f"prompt {p}: {len(ids)} tokens; next token after the prompt: {tok.decode([top1[-1]])!r}")
        if args.generate:
            print(f"  greedy: {tok.decode(ids + out[f'prompt{p}.gen_tokens'].tolist())!r}")
    print(f"smallest gap between the k-th and (k+1)-th router probability: {gap:.2e}")
    with open(args.out, "wb") as f:
        f.write(serialize(out))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes, {len(out)} tensors)")


if __name__ == "__main__":
    main()
