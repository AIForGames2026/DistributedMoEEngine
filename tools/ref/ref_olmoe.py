#!/usr/bin/env python3
"""TASK-10 reference for OLMoE-1B-7B: runs Hugging Face's own OlmoeForCausalLM
(float32, eager attention, CPU) on the weights in a GGUF file, dequantized with
llama.cpp's gguf-py, and writes golden values for the engine's forward pass:
per prompt, the token ids, every position's logits, and each layer's output
hidden state and top-k routing. Testing only.

So the comparison tests the engine's code and not quantization, the reference
uses exactly the GGUF's weights. To fit in memory, the 6.4B expert parameters
stay quantized: each expert's gate/up/down Linear dequantizes its Q8_0 rows
from the memory-mapped file when called. Everything else (attention with QK
norm, rotary embeddings, routing, combining) is transformers' code.

Usage:
  python3 tools/ref/ref_olmoe.py --gguf MODEL.gguf --config config.json \\
      --tokenizer tokenizer.json --out olmoe_golden.tfref [--generate 24]
  python3 tools/ref/ref_olmoe.py --tiny [--check]
      writes tests/golden/olmoe_tiny.{gguf,tfref}: a tiny OLMoE with random
      weights (seeded) in Q8_0, and transformers' outputs for it, for the
      engine's unit tests and CI.
"""
import argparse
import os
import sys

import numpy as np
import torch
import gguf
from tokenizers import Tokenizer
from transformers import OlmoeConfig, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402

# Golden prompts: plain text, tokenized with the model's own tokenizer, no BOS
# (tokenizer_config.json: add_bos_token false).
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Mixture-of-experts models route each token to a few experts, so",
]

# GGUF tensor name -> Hugging Face parameter name (non-expert weights).
def hf_name(name):
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    parts = name.split(".")
    if parts[0] != "blk":
        return None
    l, rest = parts[1], ".".join(parts[2:])
    table = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }
    return f"model.layers.{l}.{table[rest]}" if rest in table else None


def dequant(t):
    """Tensor in numpy order (outermost first), float32."""
    shape = tuple(int(d) for d in reversed(t.shape))
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(t.data, dtype=np.float32).reshape(shape)
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(shape).astype(np.float32)


class LazyExpertLinear(torch.nn.Module):
    """One expert's projection, dequantized from the stacked GGUF tensor on use."""

    def __init__(self, tensor, expert):
        super().__init__()
        self.tensor, self.expert = tensor, expert

    def forward(self, x):
        rows = int(self.tensor.shape[1])  # ggml dims: [row_len, rows, experts]
        if x.shape[0] == 0:  # transformers calls every expert, even ones no token chose
            return x.new_zeros((0, rows))
        w = gguf.quants.dequantize(np.asarray(self.tensor.data[self.expert]), self.tensor.tensor_type)
        return torch.nn.functional.linear(x, torch.from_numpy(w.astype(np.float32)))


def build(gguf_path, config_path):
    cfg = OlmoeConfig.from_json_file(config_path)
    cfg.torch_dtype = torch.float32
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = OlmoeForCausalLM(cfg)
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


def record(model, cfg, prompts):
    """Golden values for each token-id list in `prompts`."""
    out = {"config": torch.tensor([len(prompts), cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok,
                                   cfg.hidden_size, cfg.vocab_size], dtype=torch.int32)}
    k = cfg.num_experts_per_tok
    min_gap = float("inf")
    with torch.no_grad():
        for p, ids in enumerate(prompts):
            res = model(torch.tensor([ids]), output_hidden_states=True, output_router_logits=True, use_cache=False)
            out[f"prompt{p}.tokens"] = torch.tensor(ids, dtype=torch.int32)
            out[f"prompt{p}.logits"] = res.logits[0].contiguous()
            for l in range(cfg.num_hidden_layers):
                # hidden_states[0] is the embedding; [l + 1] is layer l's output,
                # except the last, which transformers returns after the final norm.
                if l + 1 < cfg.num_hidden_layers:
                    out[f"prompt{p}.layer{l}.hidden"] = res.hidden_states[l + 1][0].contiguous()
                probs = torch.softmax(res.router_logits[l].float(), dim=-1)
                w, idx = torch.topk(probs, k + 1, dim=-1)
                min_gap = min(min_gap, float((w[:, k - 1] - w[:, k]).min()))
                out[f"prompt{p}.layer{l}.topk_idx"] = idx[:, :k].to(torch.int32).contiguous()
                out[f"prompt{p}.layer{l}.topk_w"] = w[:, :k].contiguous()
    return out, min_gap


TINY = dict(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, num_experts=8, num_experts_per_tok=2, rms_norm_eps=1e-5, rope_theta=10000.0,
            norm_topk_prob=False, tie_word_embeddings=False, max_position_embeddings=64)
TINY_SEED = 20261106
TINY_PROMPTS = [[3, 17, 200, 45, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [250, 128, 64, 32, 16, 8, 4]]
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir):
    """A tiny OLMoE with random weights: every 2-D weight in Q8_0 (the
    reference uses the dequantized values), norms and router in f32."""
    cfg = OlmoeConfig(**TINY)
    cfg._attn_implementation = "eager"
    torch.manual_seed(TINY_SEED)
    model = OlmoeForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8 = gguf.GGMLQuantizationType.Q8_0

    def q8(x):
        """Q8_0 round trip; returns (bytes for the file, values the model uses)."""
        q = gguf.quants.quantize(x.numpy(), Q8)
        return q, torch.from_numpy(gguf.quants.dequantize(q, Q8).reshape(x.shape).astype(np.float32))

    w = gguf.GGUFWriter(path=None, arch="olmoe")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_expert_count(cfg.num_experts)
    w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_expert_feed_forward_length(cfg.intermediate_size)
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

    d, f, e, v = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.vocab_size
    sd = {}
    emb = torch.randn(v, d, generator=g)
    q, sd["model.embed_tokens.weight"] = q8(emb)
    w.add_tensor("token_embd.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    for l in range(cfg.num_hidden_layers):
        pre = f"model.layers.{l}."
        sd[pre + "input_layernorm.weight"] = vec(f"blk.{l}.attn_norm.weight", d)
        for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            sd[pre + f"self_attn.{hf}.weight"] = linear(f"blk.{l}.{gg}.weight", (d, d))
        sd[pre + "self_attn.q_norm.weight"] = vec(f"blk.{l}.attn_q_norm.weight", d)
        sd[pre + "self_attn.k_norm.weight"] = vec(f"blk.{l}.attn_k_norm.weight", d)
        sd[pre + "post_attention_layernorm.weight"] = vec(f"blk.{l}.ffn_norm.weight", d)
        # Router in f32, scaled up so the top-k choice is clear-cut.
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

    path = os.path.join(out_dir, "olmoe_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, TINY_PROMPTS)
    with open(os.path.join(out_dir, "olmoe_tiny.tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    names = ("olmoe_tiny.gguf", "olmoe_tiny.tfref")
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
    print("OK: olmoe_tiny.gguf and olmoe_tiny.tfref match a fresh run")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0, help="also print a greedy continuation (sanity check)")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, cfg = build(args.gguf, args.config)
    tok = Tokenizer.from_file(args.tokenizer)
    prompts = [tok.encode(text).ids for text in PROMPTS]
    out, _ = record(model, cfg, prompts)
    with torch.no_grad():
        for p, ids in enumerate(prompts):
            top1 = out[f"prompt{p}.logits"].argmax(-1).tolist()
            print(f"prompt {p}: {len(ids)} tokens; next token after the prompt: {tok.decode([top1[-1]])!r}")
            if args.generate:
                g = model.generate(torch.tensor([ids]), max_new_tokens=args.generate, do_sample=False)
                print(f"  greedy: {tok.decode(g[0].tolist())!r}")
    with open(args.out, "wb") as f:
        f.write(serialize(out))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes, {len(out)} tensors)")


if __name__ == "__main__":
    main()
