#!/usr/bin/env python3
"""TASK-42 reference for Qwen2-MoE and Qwen1.5-MoE: runs Hugging Face's own
Qwen2MoeForCausalLM (float32, eager attention, CPU) on the weights in a GGUF
file and writes the same golden values as ref_olmoe.py. Testing only.

The family has biases on the query, key and value projections, no QK norm, and
a shared expert scaled by a sigmoid gate; the top-k router probabilities are
used as they are (norm_topk_prob false). The experts stay quantized in the
memory-mapped file and dequantize when a token picks them.

Usage:
  python3 tools/ref/ref_qwen2moe.py --gguf MODEL.gguf --config config.json \\
      --tokenizer tokenizer.json --out qwen2moe_golden.tfref [--generate 32] [--threads N]
  python3 tools/ref/ref_qwen2moe.py --tiny [--check]
      writes tests/golden/qwen2moe_tiny.{gguf,tfref}: a tiny Qwen2-MoE with
      random weights (seeded), and transformers' outputs for it.
"""
import argparse
import os
import sys

import numpy as np
import torch
import gguf
from tokenizers import Tokenizer
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS, LazyExpertLinear, dequant, record  # noqa: E402

NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}


def hf_name(name):
    top = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}
    if name in top:
        return top[name]
    parts = name.split(".")
    rest = ".".join(parts[2:])
    return f"model.layers.{parts[1]}.{NAMES[rest]}" if parts[0] == "blk" and rest in NAMES else None


def build(gguf_path, config_path):
    cfg = Qwen2MoeConfig.from_json_file(config_path)
    cfg.torch_dtype = torch.float32
    cfg._attn_implementation = "eager"
    if cfg.mlp_only_layers or cfg.decoder_sparse_step != 1:
        sys.exit("dense layers among the MoE ones are not handled")
    with torch.device("meta"):
        model = Qwen2MoeForCausalLM(cfg)
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    state = {}
    for t in reader.tensors:
        n = hf_name(t.name)
        if n is None:
            continue
        w = dequant(t)
        if n.endswith("shared_expert_gate.weight"):
            w = w.reshape(1, -1)
        state[n] = torch.from_numpy(np.ascontiguousarray(w))
    missing = model.load_state_dict(state, strict=False, assign=True)
    for l, layer in enumerate(model.model.layers):
        for e, mlp in enumerate(layer.mlp.experts):
            mlp.gate_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_gate_exps.weight"], e)
            mlp.up_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_up_exps.weight"], e)
            mlp.down_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_down_exps.weight"], e)
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    return model.eval(), cfg


TINY = dict(vocab_size=256, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
            shared_expert_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_experts=8, num_experts_per_tok=2, norm_topk_prob=False, decoder_sparse_step=1, mlp_only_layers=[],
            rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=False, max_position_embeddings=64,
            use_sliding_window=False)
TINY_SEED = 20261120
TINY_PROMPTS = [[3, 17, 200, 45, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [250, 128, 64, 32, 16, 8, 4]]
TINY_GENERATE = 16
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir):
    """A tiny Qwen2-MoE with random weights, written as llama.cpp's converter writes
    arch "qwen2moe": Q8_0 matrices, f32 norms, biases and gates."""
    cfg = Qwen2MoeConfig(**TINY)
    cfg._attn_implementation = "eager"
    torch.manual_seed(TINY_SEED)
    model = Qwen2MoeForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8 = gguf.GGMLQuantizationType.Q8_0

    def q8(x):
        q = gguf.quants.quantize(x.numpy(), Q8)
        return q, torch.from_numpy(gguf.quants.dequantize(q, Q8).reshape(x.shape).astype(np.float32))

    d, e, v = cfg.hidden_size, cfg.num_experts, cfg.vocab_size
    f, fs = cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size
    hq, hk = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = d // hq
    w = gguf.GGUFWriter(path=None, arch="qwen2moe")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(d)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_expert_feed_forward_length(f)
    w.add_expert_shared_feed_forward_length(fs)
    w.add_head_count(hq)
    w.add_head_count_kv(hk)
    w.add_expert_count(e)
    w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(cfg.rope_theta)

    def linear(name, shape):
        q, deq = q8(torch.randn(*shape, generator=g) / shape[-1] ** 0.5)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q8)
        return deq

    def vec(name, n, base=1.0, scale=0.1):
        x = base + torch.randn(n, generator=g) * scale
        w.add_tensor(name, x.numpy())
        return x

    sd = {}
    q, sd["model.embed_tokens.weight"] = q8(torch.randn(v, d, generator=g))
    w.add_tensor("token_embd.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    for l in range(cfg.num_hidden_layers):
        pre = f"model.layers.{l}."
        sd[pre + "input_layernorm.weight"] = vec(f"blk.{l}.attn_norm.weight", d)
        for hf, gg, rows in (("q", "attn_q", hq * hd), ("k", "attn_k", hk * hd), ("v", "attn_v", hk * hd)):
            sd[pre + f"self_attn.{hf}_proj.weight"] = linear(f"blk.{l}.{gg}.weight", (rows, d))
            sd[pre + f"self_attn.{hf}_proj.bias"] = vec(f"blk.{l}.{gg}.bias", rows, base=0.0, scale=0.5)
        sd[pre + "self_attn.o_proj.weight"] = linear(f"blk.{l}.attn_output.weight", (d, hq * hd))
        sd[pre + "post_attention_layernorm.weight"] = vec(f"blk.{l}.ffn_norm.weight", d)
        router = torch.randn(e, d, generator=g) * (4.0 / d ** 0.5)
        w.add_tensor(f"blk.{l}.ffn_gate_inp.weight", router.numpy())
        sd[pre + "mlp.gate.weight"] = router
        for hf, gg, shape in (("gate_proj", "ffn_gate_exps", (f, d)), ("up_proj", "ffn_up_exps", (f, d)),
                              ("down_proj", "ffn_down_exps", (d, f))):
            q, deq = q8(torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5)
            w.add_tensor(f"blk.{l}.{gg}.weight", q, raw_shape=q.shape, raw_dtype=Q8)
            for x in range(e):
                sd[pre + f"mlp.experts.{x}.{hf}.weight"] = deq[x]
        sd[pre + "mlp.shared_expert.gate_proj.weight"] = linear(f"blk.{l}.ffn_gate_shexp.weight", (fs, d))
        sd[pre + "mlp.shared_expert.up_proj.weight"] = linear(f"blk.{l}.ffn_up_shexp.weight", (fs, d))
        sd[pre + "mlp.shared_expert.down_proj.weight"] = linear(f"blk.{l}.ffn_down_shexp.weight", (d, fs))
        sd[pre + "mlp.shared_expert_gate.weight"] = vec(f"blk.{l}.ffn_gate_inp_shexp.weight", d, base=0.0, scale=0.5).reshape(1, d)
    sd["model.norm.weight"] = vec("output_norm.weight", d)
    q, sd["lm_head.weight"] = q8(torch.randn(v, d, generator=g) / d ** 0.5)
    w.add_tensor("output.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    missing = model.load_state_dict(sd, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    path = os.path.join(out_dir, "qwen2moe_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, TINY_PROMPTS, TINY_GENERATE)
    with open(os.path.join(out_dir, "qwen2moe_tiny.tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    names = ("qwen2moe_tiny.gguf", "qwen2moe_tiny.tfref")
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
    print("OK: qwen2moe_tiny.gguf and qwen2moe_tiny.tfref match a fresh run")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0, help="also record this many tokens of greedy generate()")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    model, cfg = build(args.gguf, args.config)
    tok = Tokenizer.from_file(args.tokenizer)
    # Qwen's tokenizer adds no BOS, and neither does the engine's.
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
