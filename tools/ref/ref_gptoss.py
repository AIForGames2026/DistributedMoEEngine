#!/usr/bin/env python3
"""TASK-39 reference for gpt-oss: runs Hugging Face's own GptOssForCausalLM
(float32, eager attention, CPU) on the weights in a llama.cpp GGUF file and
writes golden values like ref_olmoe.py. Testing only.

llama.cpp's converter splits each expert's interleaved gate_up projection into
gate and up and transposes it; loading here interleaves them back. The experts
stay quantized (MXFP4) in the memory-mapped file: `LazyExperts` is transformers'
own CPU loop over the experts a forward pass hit, with each expert dequantized
when used.

Usage:
  python3 tools/ref/ref_gptoss.py --gguf MODEL.gguf --config config.json \\
      --tokenizer tokenizer.json --out gptoss_golden.tfref [--generate 32] [--threads N]
  python3 tools/ref/ref_gptoss.py --tiny [--check]
      writes tests/golden/gptoss_tiny.{gguf,tfref}: a tiny gpt-oss with sinks,
      a 4-position sliding window, biases, and YaRN, random weights (seeded),
      and transformers' outputs.
"""
import argparse
import os
import sys

import numpy as np
import torch
import gguf
from tokenizers import Tokenizer
from transformers import GptOssConfig, GptOssForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS, dequant, record  # noqa: E402

NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_sinks.weight": "self_attn.sinks",
    "ffn_gate_inp.weight": "mlp.router.weight",
    "ffn_gate_inp.bias": "mlp.router.bias",
}
for _p, _h in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")):
    NAMES[f"{_p}.weight"] = f"self_attn.{_h}.weight"
    NAMES[f"{_p}.bias"] = f"self_attn.{_h}.bias"


def hf_name(name):
    top = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}
    if name in top:
        return top[name]
    parts = name.split(".")
    rest = ".".join(parts[2:])
    return f"model.layers.{parts[1]}.{NAMES[rest]}" if parts[0] == "blk" and rest in NAMES else None


class LazyExperts(torch.nn.Module):
    """transformers' GptOssExperts CPU path, with each expert's weights dequantized when it is hit.
    `get(e)` returns that expert's gate_up (d x 2f, interleaved), its bias, down (f x d), and its bias."""

    def __init__(self, get, num_experts, hidden_size):
        super().__init__()
        self.get, self.num_experts, self.hidden_size = get, num_experts, hidden_size
        self.alpha, self.limit = 1.702, 7.0

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit[:]:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])
            gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias = self.get(int(expert_idx))
            current_state = hidden_states[token_idx]
            gate_up = current_state @ gate_up_proj + gate_up_proj_bias
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu
            out = gated_output @ down_proj + down_proj_bias
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        return next_states.view(batch_size, -1, self.hidden_size)


def interleave(gate, up):
    """gate and up (..., f rows) back into transformers' gate_up columns: gate at even, up at odd."""
    out = np.empty(gate.shape[:-1] + (2 * gate.shape[-1],), dtype=np.float32)
    out[..., ::2], out[..., 1::2] = gate, up
    return out


def expert_getter(by_name, l):
    t = {p: by_name[f"blk.{l}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down")}
    b = {p: dequant(by_name[f"blk.{l}.ffn_{p}_exps.bias"]) for p in ("gate", "up", "down")}

    def one(p, e):
        w = gguf.quants.dequantize(np.asarray(t[p].data[e]), t[p].tensor_type)
        return w.astype(np.float32).reshape(int(t[p].shape[1]), int(t[p].shape[0]))

    def get(e):
        gate_up = interleave(one("gate", e).T, one("up", e).T)
        bias = interleave(b["gate"][e], b["up"][e])
        return (torch.from_numpy(gate_up), torch.from_numpy(bias), torch.from_numpy(np.ascontiguousarray(one("down", e).T)),
                torch.from_numpy(np.ascontiguousarray(b["down"][e])))

    return get


def build(gguf_path, config_path):
    cfg = GptOssConfig.from_json_file(config_path)
    cfg.torch_dtype = torch.float32
    cfg._attn_implementation = "eager"
    # The GGUF holds dequantizable weights, not the checkpoint's MXFP4 layout.
    if hasattr(cfg, "quantization_config"):
        delattr(cfg, "quantization_config")
    cfg.num_experts = cfg.num_local_experts
    with torch.device("meta"):
        model = GptOssForCausalLM(cfg)
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    state = {}
    for t in reader.tensors:
        n = hf_name(t.name)
        if n is not None:
            state[n] = torch.from_numpy(np.ascontiguousarray(dequant(t)))
    missing = model.load_state_dict(state, strict=False, assign=True)
    for l, layer in enumerate(model.model.layers):
        layer.mlp.experts = LazyExperts(expert_getter(by_name, l), cfg.num_local_experts, cfg.hidden_size)
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    return model.eval(), cfg


TINY = dict(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, num_local_experts=8, num_experts_per_tok=2, rms_norm_eps=1e-5,
            rope_theta=150000.0, tie_word_embeddings=False, max_position_embeddings=64, sliding_window=4,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"], swiglu_limit=7.0)
TINY_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16, "beta_fast": 32.0,
             "beta_slow": 1.0, "truncate": False}
TINY_SEED = 20261120
TINY_PROMPTS = [[3, 17, 200, 45, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [250, 128, 64, 32, 16, 8, 4]]
TINY_GENERATE = 16
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir):
    """A tiny gpt-oss written as llama.cpp's converter writes it: Q8_0 matrices, MXFP4 experts,
    f32 biases, sinks and router."""
    cfg = GptOssConfig(**TINY, rope_scaling=dict(TINY_YARN))
    cfg.rope_scaling = dict(TINY_YARN)
    cfg._attn_implementation = "eager"
    cfg.num_experts = cfg.num_local_experts
    torch.manual_seed(TINY_SEED)
    model = GptOssForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8, MX = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.MXFP4

    def quant(x, kind):
        q = gguf.quants.quantize(x.numpy(), kind)
        return q, torch.from_numpy(gguf.quants.dequantize(q, kind).reshape(x.shape).astype(np.float32))

    A = "gpt-oss"
    w = gguf.GGUFWriter(path=None, arch=A)
    d, f, e, hq, hk, hd = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_attention_heads, \
        cfg.num_key_value_heads, cfg.head_dim
    for key, val in (("block_count", cfg.num_hidden_layers), ("context_length", cfg.max_position_embeddings),
                     ("embedding_length", d), ("feed_forward_length", f), ("attention.head_count", hq),
                     ("attention.head_count_kv", hk), ("attention.key_length", hd), ("attention.value_length", hd),
                     ("expert_count", e), ("expert_used_count", cfg.num_experts_per_tok),
                     ("expert_feed_forward_length", f), ("attention.sliding_window", cfg.sliding_window),
                     ("rope.scaling.original_context_length", TINY_YARN["original_max_position_embeddings"])):
        w.add_uint32(f"{A}.{key}", val)
    w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    w.add_float32(f"{A}.rope.freq_base", cfg.rope_theta)
    w.add_string(f"{A}.rope.scaling.type", "yarn")
    w.add_float32(f"{A}.rope.scaling.factor", TINY_YARN["factor"])
    w.add_float32(f"{A}.rope.scaling.yarn_beta_fast", TINY_YARN["beta_fast"])
    w.add_float32(f"{A}.rope.scaling.yarn_beta_slow", TINY_YARN["beta_slow"])

    def linear(name, shape):
        x = torch.randn(*shape, generator=g) / shape[-1] ** 0.5
        q, deq = quant(x, Q8)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q8)
        return deq

    def vec(name, n, base=1.0, scale=0.1):
        x = base + torch.randn(n, generator=g) * scale
        w.add_tensor(name, x.numpy())
        return x

    sd = {}
    q, sd["model.embed_tokens.weight"] = quant(torch.randn(cfg.vocab_size, d, generator=g), Q8)
    w.add_tensor("token_embd.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    for l in range(cfg.num_hidden_layers):
        pre, b = f"model.layers.{l}.", f"blk.{l}."
        sd[pre + "input_layernorm.weight"] = vec(b + "attn_norm.weight", d)
        for gg, hf, rows, cols in (("attn_q", "q_proj", hq * hd, d), ("attn_k", "k_proj", hk * hd, d),
                                   ("attn_v", "v_proj", hk * hd, d), ("attn_output", "o_proj", d, hq * hd)):
            sd[pre + f"self_attn.{hf}.weight"] = linear(b + f"{gg}.weight", (rows, cols))
            sd[pre + f"self_attn.{hf}.bias"] = vec(b + f"{gg}.bias", rows, 0.0, 0.2)
        sd[pre + "self_attn.sinks"] = vec(b + "attn_sinks.weight", hq, 0.0, 1.0)
        sd[pre + "post_attention_layernorm.weight"] = vec(b + "post_attention_norm.weight", d)
        router = torch.randn(e, d, generator=g) * (4.0 / d ** 0.5)
        w.add_tensor(b + "ffn_gate_inp.weight", router.numpy())
        sd[pre + "mlp.router.weight"] = router
        sd[pre + "mlp.router.bias"] = vec(b + "ffn_gate_inp.bias", e, 0.0, 0.5)
        mats = {}
        for p, shape in (("gate", (f, d)), ("up", (f, d)), ("down", (d, f))):
            stacked = torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5
            q, mats[p] = quant(stacked, MX)
            w.add_tensor(b + f"ffn_{p}_exps.weight", q, raw_shape=q.shape, raw_dtype=MX)
            bias = torch.randn(e, shape[0], generator=g) * 0.2
            w.add_tensor(b + f"ffn_{p}_exps.bias", bias.numpy())
            mats[p + "_b"] = bias
        sd[pre + "mlp.experts.gate_up_proj"] = torch.from_numpy(interleave(mats["gate"].transpose(1, 2).numpy(),
                                                                           mats["up"].transpose(1, 2).numpy()))
        sd[pre + "mlp.experts.gate_up_proj_bias"] = torch.from_numpy(interleave(mats["gate_b"].numpy(), mats["up_b"].numpy()))
        sd[pre + "mlp.experts.down_proj"] = mats["down"].transpose(1, 2).contiguous()
        sd[pre + "mlp.experts.down_proj_bias"] = mats["down_b"]
    sd["model.norm.weight"] = vec("output_norm.weight", d)
    q, sd["lm_head.weight"] = quant(torch.randn(cfg.vocab_size, d, generator=g) / d ** 0.5, Q8)
    w.add_tensor("output.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    missing = model.load_state_dict(sd, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    path = os.path.join(out_dir, "gptoss_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, TINY_PROMPTS, TINY_GENERATE)
    with open(os.path.join(out_dir, "gptoss_tiny.tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    names = ("gptoss_tiny.gguf", "gptoss_tiny.tfref")
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
    print("OK: gptoss_tiny.gguf and gptoss_tiny.tfref match a fresh run")


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
