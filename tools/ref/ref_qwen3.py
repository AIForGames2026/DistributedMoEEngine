#!/usr/bin/env python3
"""TASK-12 reference for Qwen3-30B-A3B: runs Hugging Face's own
Qwen3MoeForCausalLM (float32, eager attention, CPU) on the weights in a GGUF
file and writes the same golden values as ref_olmoe.py: per prompt, the token
ids, every position's logits, each layer's output hidden state and top-k
routing, and optionally greedy generate(). Testing only.

As in ref_olmoe.py, the experts stay quantized and dequantize their rows from
the memory-mapped file when called, so the 30B model needs about 6 GB of RAM
for its float32 non-expert weights plus the mapped file. `--yarn F` turns on
transformers' YaRN rope scaling with factor F over the model's 32,768-token
native context, as Qwen documents for 128k (F = 4).

Usage:
  python3 tools/ref/ref_qwen3.py --gguf Qwen3-30B-A3B-Q8_0.gguf --config config.json \\
      --tokenizer tokenizer.json --out qwen3_golden.tfref [--yarn 4] [--generate 32]
  python3 tools/ref/ref_qwen3.py --tiny [--check]
      writes tests/golden/qwen3_tiny.{gguf,tfref}: a tiny Qwen3-MoE with
      random weights (seeded) and YaRN, and transformers' outputs for it, for
      the engine's unit tests and CI.
"""
import argparse
import os
import sys

import torch
import gguf
from tokenizers import Tokenizer
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS, LazyExpertLinear, dequant, hf_name, record  # noqa: E402


def build(gguf_path, config_path, yarn):
    cfg = Qwen3MoeConfig.from_json_file(config_path)
    cfg.torch_dtype = torch.float32
    cfg._attn_implementation = "eager"
    if yarn:
        cfg.rope_scaling = {"rope_type": "yarn", "factor": float(yarn), "original_max_position_embeddings": 32768}
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(cfg)
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    state = {}
    for t in reader.tensors:
        n = hf_name(t.name)
        if n is not None:
            state[n] = torch.from_numpy(dequant(t))
    missing = model.load_state_dict(state, strict=False, assign=True)
    for l, layer in enumerate(model.model.layers):
        for e, mlp in enumerate(layer.mlp.experts):
            mlp.gate_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_gate_exps.weight"], e)
            mlp.up_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_up_exps.weight"], e)
            mlp.down_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_down_exps.weight"], e)
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    # Rotary inverse frequencies are buffers built on the meta device; rebuild them.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    return model.eval(), cfg


TINY = dict(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2,
            norm_topk_prob=True, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
            max_position_embeddings=64, decoder_sparse_step=1, mlp_only_layers=[])
TINY_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}
TINY_SEED = 20261111
TINY_PROMPTS = [[3, 17, 200, 45, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [250, 128, 64, 32, 16, 8, 4]]
TINY_GENERATE = 16
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir):
    """A tiny Qwen3-MoE with random weights and YaRN (factor 4 over 16
    positions): grouped KV heads, head_dim unlike hidden / heads, and every
    2-D weight in Q8_0 (the reference uses the dequantized values)."""
    import numpy as np

    cfg = Qwen3MoeConfig(**TINY)
    cfg.rope_scaling = dict(TINY_YARN)
    cfg._attn_implementation = "eager"
    torch.manual_seed(TINY_SEED)
    model = Qwen3MoeForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8 = gguf.GGMLQuantizationType.Q8_0

    def q8(x):
        q = gguf.quants.quantize(x.numpy(), Q8)
        return q, torch.from_numpy(gguf.quants.dequantize(q, Q8).reshape(x.shape).astype(np.float32))

    w = gguf.GGUFWriter(path=None, arch="qwen3moe")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_key_length(cfg.head_dim)
    w.add_value_length(cfg.head_dim)
    w.add_expert_count(cfg.num_experts)
    w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_expert_feed_forward_length(cfg.moe_intermediate_size)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(cfg.rope_theta)

    def linear(name, shape):
        x = torch.randn(*shape, generator=g) / shape[-1] ** 0.5
        q, deq = q8(x)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q8)
        return deq

    def vec(name, n, scale=0.1):
        x = 1.0 + torch.randn(n, generator=g) * scale
        w.add_tensor(name, x.numpy())
        return x

    d, f, e, v = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.vocab_size
    qd, kvd, hd = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim, cfg.head_dim
    sd = {}
    q, sd["model.embed_tokens.weight"] = q8(torch.randn(v, d, generator=g))
    w.add_tensor("token_embd.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    for l in range(cfg.num_hidden_layers):
        pre = f"model.layers.{l}."
        sd[pre + "input_layernorm.weight"] = vec(f"blk.{l}.attn_norm.weight", d)
        for hf, gg, shape in (("q_proj", "attn_q", (qd, d)), ("k_proj", "attn_k", (kvd, d)),
                              ("v_proj", "attn_v", (kvd, d)), ("o_proj", "attn_output", (d, qd))):
            sd[pre + f"self_attn.{hf}.weight"] = linear(f"blk.{l}.{gg}.weight", shape)
        sd[pre + "self_attn.q_norm.weight"] = vec(f"blk.{l}.attn_q_norm.weight", hd)
        sd[pre + "self_attn.k_norm.weight"] = vec(f"blk.{l}.attn_k_norm.weight", hd)
        sd[pre + "post_attention_layernorm.weight"] = vec(f"blk.{l}.ffn_norm.weight", d)
        router = torch.randn(e, d, generator=g) * (4.0 / d ** 0.5)
        w.add_tensor(f"blk.{l}.ffn_gate_inp.weight", router.numpy())
        sd[pre + "mlp.gate.weight"] = router
        for hf, gg, shape in (("gate_proj", "ffn_gate_exps", (f, d)), ("up_proj", "ffn_up_exps", (f, d)),
                              ("down_proj", "ffn_down_exps", (d, f))):
            stacked = torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5
            q, deq = q8(stacked)
            w.add_tensor(f"blk.{l}.{gg}.weight", q, raw_shape=q.shape, raw_dtype=Q8)
            for x in range(e):
                sd[pre + f"mlp.experts.{x}.{hf}.weight"] = deq[x]
    sd["model.norm.weight"] = vec("output_norm.weight", d)
    q, sd["lm_head.weight"] = q8(torch.randn(v, d, generator=g) / d ** 0.5)
    w.add_tensor("output.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    missing = model.load_state_dict(sd, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    path = os.path.join(out_dir, "qwen3_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, TINY_PROMPTS, TINY_GENERATE)
    with open(os.path.join(out_dir, "qwen3_tiny.tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    names = ("qwen3_tiny.gguf", "qwen3_tiny.tfref")
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
    print("OK: qwen3_tiny.gguf and qwen3_tiny.tfref match a fresh run")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yarn", type=float, default=0, help="YaRN factor over 32,768 positions (Qwen uses 4 for 128k)")
    ap.add_argument("--generate", type=int, default=0, help="also record this many tokens of greedy generate()")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    model, cfg = build(args.gguf, args.config, args.yarn)
    tok = Tokenizer.from_file(args.tokenizer)
    prompts = [tok.encode(text).ids for text in PROMPTS]
    out, gap = record(model, cfg, prompts, args.generate)
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
