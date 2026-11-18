#!/usr/bin/env python3
"""TASK-39 reference for DeepSeek V3's architecture (Moonlight-16B-A3B, Kimi K2):
runs Hugging Face's own DeepseekV3ForCausalLM (float32, eager attention, CPU) on
the weights in a llama.cpp GGUF file (architecture "deepseek2") and writes golden
values like ref_glm53.py. Testing only.

Loading undoes llama.cpp's converter: each head's key and value projections were
split out of kv_b_proj, the key part transposed, and are joined back here. The
routed experts stay quantized in the memory-mapped file and dequantize when a
token picks them.

Usage:
  python3 tools/ref/ref_deepseek.py --gguf MODEL.gguf --config config.json \\
      --tokenizer DIR --out deepseek_golden.tfref [--generate 32] [--threads N]
      (DIR holds Moonshot's tokenizer files; they load with trust_remote_code)
  python3 tools/ref/ref_deepseek.py --tiny [--check]
      writes tests/golden/deepseek_tiny.{gguf,tfref}: a tiny model with a query
      rank, four expert groups of which two are used, a choice bias, a dense
      first layer, and YaRN, with random weights (seeded), and transformers' outputs.
"""
import argparse
import os
import sys

import numpy as np
import torch
import gguf
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS, LazyExpertLinear, dequant  # noqa: E402

NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
}


def hf_name(name):
    top = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}
    if name in top:
        return top[name]
    parts = name.split(".")
    rest = ".".join(parts[2:])
    return f"model.layers.{parts[1]}.{NAMES[rest]}" if parts[0] == "blk" and rest in NAMES else None


def split_kv_b(kv_b, heads, nope, v):
    """llama.cpp's split of kv_b_proj into k_b (transposed) and v_b, per head."""
    kv = kv_b.reshape(heads, nope + v, -1)
    return np.ascontiguousarray(kv[:, :nope, :].transpose(0, 2, 1)), np.ascontiguousarray(kv[:, nope:, :])


def join_kv_b(k_b, v_b):
    heads = k_b.shape[0]
    return np.concatenate([k_b.transpose(0, 2, 1), v_b], axis=1).reshape(-1, k_b.shape[1])


def build(gguf_path, config_path):
    cfg = DeepseekV3Config.from_json_file(config_path)
    cfg.torch_dtype = torch.float32
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(cfg)
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    state = {}
    for t in reader.tensors:
        n = hf_name(t.name)
        if n is not None:
            state[n] = torch.from_numpy(np.ascontiguousarray(dequant(t)))
    for l in range(cfg.num_hidden_layers):
        k_b = dequant(by_name[f"blk.{l}.attn_k_b.weight"])
        v_b = dequant(by_name[f"blk.{l}.attn_v_b.weight"])
        state[f"model.layers.{l}.self_attn.kv_b_proj.weight"] = torch.from_numpy(join_kv_b(k_b, v_b))
    missing = model.load_state_dict(state, strict=False, assign=True)
    for l, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "experts"):
            continue
        for e, mlp in enumerate(layer.mlp.experts):
            mlp.gate_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_gate_exps.weight"], e)
            mlp.up_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_up_exps.weight"], e)
            mlp.down_proj = LazyExpertLinear(by_name[f"blk.{l}.ffn_down_exps.weight"], e)
        if f"model.layers.{l}.mlp.gate.e_score_correction_bias" not in state:
            layer.mlp.gate.e_score_correction_bias = torch.zeros(cfg.n_routed_experts)
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k and not k.endswith("e_score_correction_bias")]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    return model.eval(), cfg


def choice_gap(x, gate, k):
    """The smallest gap, over rows, between the k-th and (k+1)-th choice score (sigmoid plus bias),
    among the experts the group limit leaves."""
    logits = torch.nn.functional.linear(x.reshape(-1, x.shape[-1]).float(), gate.weight.float())
    choice = torch.sigmoid(logits) + gate.e_score_correction_bias
    if gate.topk_group < gate.n_group:
        e = choice.shape[-1]
        grouped = choice.view(-1, gate.n_group, e // gate.n_group)
        gidx = torch.topk(grouped.topk(2, dim=-1)[0].sum(-1), gate.topk_group, dim=-1)[1]
        mask = torch.zeros(grouped.shape[:2]).scatter_(1, gidx, 1).bool()
        choice = choice.masked_fill(~mask.unsqueeze(-1).expand_as(grouped).reshape(-1, e), 0.0)
    s = torch.topk(choice, k + 1, dim=-1).values
    return float((s[..., k - 1] - s[..., k]).min())


def record(model, cfg, prompts, gen):
    """Golden values for each token-id list in `prompts`, plus `gen` tokens of greedy generate()."""
    layers = model.model.layers
    sparse = [l for l in range(cfg.num_hidden_layers) if hasattr(layers[l].mlp, "experts")]
    out = {"config": torch.tensor([len(prompts), cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok,
                                   cfg.hidden_size, cfg.vocab_size], dtype=torch.int32)}
    k = cfg.num_experts_per_tok
    min_gap = float("inf")
    seen = {}

    def on_gate(l):
        def hook(mod, args, res):
            if seen.get("on"):
                seen[("gate", l)] = (args[0].clone(), res[0].clone(), res[1].clone())
        return hook

    def on_layer(l):
        def hook(_mod, _args, res):
            if seen.get("on"):
                seen[("hidden", l)] = (res[0] if isinstance(res, tuple) else res)[0].clone()
        return hook

    hooks = [layers[l].mlp.gate.register_forward_hook(on_gate(l)) for l in sparse]
    hooks += [layers[l].register_forward_hook(on_layer(l)) for l in range(cfg.num_hidden_layers)]
    with torch.no_grad():
        for p, ids in enumerate(prompts):
            seen.clear()
            seen["on"] = True
            res = model(torch.tensor([ids]), use_cache=False)
            seen["on"] = False
            out[f"prompt{p}.tokens"] = torch.tensor(ids, dtype=torch.int32)
            out[f"prompt{p}.logits"] = res.logits[0].contiguous()
            for l in range(cfg.num_hidden_layers):
                out[f"prompt{p}.layer{l}.hidden"] = seen[("hidden", l)].contiguous()
                if l in sparse:
                    x, idx, w = seen[("gate", l)]
                    min_gap = min(min_gap, choice_gap(x, layers[l].mlp.gate, k))
                    order = torch.argsort(idx, dim=-1)
                    out[f"prompt{p}.layer{l}.topk_idx"] = torch.gather(idx, -1, order).to(torch.int32).contiguous()
                    out[f"prompt{p}.layer{l}.topk_w"] = torch.gather(w, -1, order).contiguous()
            if gen:
                gaps = []

                def gap(mod, args, _res):
                    g_now = choice_gap(args[0][:, -1:], mod, k)
                    if mod is layers[sparse[0]].mlp.gate:
                        gaps.append(g_now)
                    else:
                        gaps[-1] = min(gaps[-1], g_now)

                gate_hooks = [layers[l].mlp.gate.register_forward_hook(gap) for l in sparse]
                g = model.generate(torch.tensor([ids]), max_new_tokens=gen, min_new_tokens=gen, do_sample=False,
                                   eos_token_id=None, pad_token_id=0, output_logits=True, return_dict_in_generate=True)
                for h in gate_hooks:
                    h.remove()
                out[f"prompt{p}.gen_tokens"] = g.sequences[0, len(ids):].to(torch.int32).contiguous()
                out[f"prompt{p}.gen_logits"] = torch.cat([x.float() for x in g.logits]).contiguous()
                if len(gaps) != gen:
                    sys.exit(f"recorded {len(gaps)} router gaps for {gen} generated steps")
                out[f"prompt{p}.gen_gaps"] = torch.tensor(gaps, dtype=torch.float32)
    for h in hooks:
        h.remove()
    return out, min_gap


TINY = dict(vocab_size=256, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=4, q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=32,
            qk_rope_head_dim=8, v_head_dim=24, n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=3,
            n_group=4, topk_group=2, first_k_dense_replace=1, norm_topk_prob=True, routed_scaling_factor=2.5,
            rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False, max_position_embeddings=64,
            rope_interleave=True)
TINY_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16, "mscale": 1.0,
             "mscale_all_dim": 1.0, "beta_fast": 32.0, "beta_slow": 1.0}
TINY_SEED = 20261119
TINY_PROMPTS = [[3, 17, 200, 45, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [250, 128, 64, 32, 16, 8, 4]]
TINY_GENERATE = 16
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir):
    """The tiny model, written as llama.cpp's converter writes DeepSeek V3, Q8_0 matrices."""
    cfg = DeepseekV3Config(**TINY)
    cfg.rope_scaling = dict(TINY_YARN)
    cfg._attn_implementation = "eager"
    torch.manual_seed(TINY_SEED)
    model = DeepseekV3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8 = gguf.GGMLQuantizationType.Q8_0

    def q8(x):
        q = gguf.quants.quantize(x.numpy(), Q8)
        return q, torch.from_numpy(gguf.quants.dequantize(q, Q8).reshape(x.shape).astype(np.float32))

    A = "deepseek2"
    w = gguf.GGUFWriter(path=None, arch=A)
    d, heads = cfg.hidden_size, cfg.num_attention_heads
    nope, rope, v, kvr, qr = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank, cfg.q_lora_rank
    f, fd, e = cfg.moe_intermediate_size, cfg.intermediate_size, cfg.n_routed_experts
    for key, val in (("block_count", cfg.num_hidden_layers), ("context_length", cfg.max_position_embeddings),
                     ("embedding_length", d), ("feed_forward_length", fd), ("attention.head_count", heads),
                     ("attention.head_count_kv", 1), ("attention.q_lora_rank", qr), ("attention.kv_lora_rank", kvr),
                     ("attention.key_length", kvr + rope), ("attention.value_length", kvr),
                     ("attention.key_length_mla", nope + rope), ("attention.value_length_mla", v),
                     ("rope.dimension_count", rope), ("expert_count", e), ("expert_used_count", cfg.num_experts_per_tok),
                     ("expert_shared_count", cfg.n_shared_experts), ("expert_feed_forward_length", f),
                     ("leading_dense_block_count", cfg.first_k_dense_replace), ("expert_gating_func", 2),
                     ("expert_group_count", cfg.n_group), ("expert_group_used_count", cfg.topk_group),
                     ("rope.scaling.original_context_length", TINY_YARN["original_max_position_embeddings"])):
        w.add_uint32(f"{A}.{key}", val)
    w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    w.add_float32(f"{A}.rope.freq_base", cfg.rope_theta)
    w.add_float32(f"{A}.expert_weights_scale", cfg.routed_scaling_factor)
    w.add_bool(f"{A}.expert_weights_norm", cfg.norm_topk_prob)
    w.add_string(f"{A}.rope.scaling.type", "yarn")
    w.add_float32(f"{A}.rope.scaling.factor", TINY_YARN["factor"])
    w.add_float32(f"{A}.rope.scaling.yarn_log_multiplier", 0.1 * TINY_YARN["mscale_all_dim"])

    def linear(name, shape):
        x = torch.randn(*shape, generator=g) / shape[-1] ** 0.5
        q, deq = q8(x)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q8)
        return deq

    def vec(name, n, scale=0.1):
        x = 1.0 + torch.randn(n, generator=g) * scale
        w.add_tensor(name, x.numpy())
        return x

    sd = {}
    q, sd["model.embed_tokens.weight"] = q8(torch.randn(cfg.vocab_size, d, generator=g))
    w.add_tensor("token_embd.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    for l in range(cfg.num_hidden_layers):
        pre, b = f"model.layers.{l}.", f"blk.{l}."
        sd[pre + "input_layernorm.weight"] = vec(b + "attn_norm.weight", d)
        sd[pre + "self_attn.q_a_proj.weight"] = linear(b + "attn_q_a.weight", (qr, d))
        sd[pre + "self_attn.q_a_layernorm.weight"] = vec(b + "attn_q_a_norm.weight", qr)
        sd[pre + "self_attn.q_b_proj.weight"] = linear(b + "attn_q_b.weight", (heads * (nope + rope), qr))
        sd[pre + "self_attn.kv_a_proj_with_mqa.weight"] = linear(b + "attn_kv_a_mqa.weight", (kvr + rope, d))
        sd[pre + "self_attn.kv_a_layernorm.weight"] = vec(b + "attn_kv_a_norm.weight", kvr)
        # k_b and v_b are written from Q8_0 round trips of kv_b's parts, as the converter splits it.
        kv_b = torch.randn(heads * (nope + v), kvr, generator=g) / kvr ** 0.5
        k_b, v_b = split_kv_b(kv_b.numpy(), heads, nope, v)
        qk, k_deq = q8(torch.from_numpy(k_b))
        qv, v_deq = q8(torch.from_numpy(v_b))
        w.add_tensor(b + "attn_k_b.weight", qk, raw_shape=qk.shape, raw_dtype=Q8)
        w.add_tensor(b + "attn_v_b.weight", qv, raw_shape=qv.shape, raw_dtype=Q8)
        sd[pre + "self_attn.kv_b_proj.weight"] = torch.from_numpy(join_kv_b(k_deq.numpy(), v_deq.numpy()))
        sd[pre + "self_attn.o_proj.weight"] = linear(b + "attn_output.weight", (d, heads * v))
        sd[pre + "post_attention_layernorm.weight"] = vec(b + "ffn_norm.weight", d)
        if l < cfg.first_k_dense_replace:
            for hf, gg, shape in (("gate_proj", "ffn_gate", (fd, d)), ("up_proj", "ffn_up", (fd, d)),
                                  ("down_proj", "ffn_down", (d, fd))):
                sd[pre + f"mlp.{hf}.weight"] = linear(b + f"{gg}.weight", shape)
            continue
        router = torch.randn(e, d, generator=g) * (4.0 / d ** 0.5)
        w.add_tensor(b + "ffn_gate_inp.weight", router.numpy())
        sd[pre + "mlp.gate.weight"] = router
        bias = torch.randn(e, generator=g) * 0.05
        w.add_tensor(b + "exp_probs_b.bias", bias.numpy())
        sd[pre + "mlp.gate.e_score_correction_bias"] = bias
        fs = f * cfg.n_shared_experts
        for hf, gg, shape in (("gate_proj", "ffn_gate_shexp", (fs, d)), ("up_proj", "ffn_up_shexp", (fs, d)),
                              ("down_proj", "ffn_down_shexp", (d, fs))):
            sd[pre + f"mlp.shared_experts.{hf}.weight"] = linear(b + f"{gg}.weight", shape)
        for hf, gg, shape in (("gate_proj", "ffn_gate_exps", (f, d)), ("up_proj", "ffn_up_exps", (f, d)),
                              ("down_proj", "ffn_down_exps", (d, f))):
            stacked = torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5
            q, deq = q8(stacked)
            w.add_tensor(b + f"{gg}.weight", q, raw_shape=q.shape, raw_dtype=Q8)
            for x in range(e):
                sd[pre + f"mlp.experts.{x}.{hf}.weight"] = deq[x]
    sd["model.norm.weight"] = vec("output_norm.weight", d)
    q, sd["lm_head.weight"] = q8(torch.randn(cfg.vocab_size, d, generator=g) / d ** 0.5)
    w.add_tensor("output.weight", q, raw_shape=q.shape, raw_dtype=Q8)
    missing = model.load_state_dict(sd, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    path = os.path.join(out_dir, "deepseek_tiny.gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, TINY_PROMPTS, TINY_GENERATE)
    with open(os.path.join(out_dir, "deepseek_tiny.tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    names = ("deepseek_tiny.gguf", "deepseek_tiny.tfref")
    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    if not check:
        gap = write_tiny(GOLDEN_DIR)
        for n in names:
            print(f"wrote tests/golden/{n} ({os.path.getsize(os.path.join(GOLDEN_DIR, n))} bytes)")
        print(f"smallest gap between the k-th and (k+1)-th choice score: {gap:.2e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        write_tiny(tmp)
        bad = [n for n in names if digest(os.path.join(tmp, n)) != digest(os.path.join(GOLDEN_DIR, n))]
    if bad:
        sys.exit(f"MISMATCH: {', '.join(bad)} differ from a fresh run")
    print("OK: deepseek_tiny.gguf and deepseek_tiny.tfref match a fresh run")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True, help="directory with the model's tokenizer files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0, help="also record this many tokens of greedy generate()")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    model, cfg = build(args.gguf, args.config)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompts = [tok.encode(text, add_special_tokens=False) for text in PROMPTS]
    out, gap = record(model, cfg, prompts, args.generate)
    for p, ids in enumerate(prompts):
        top1 = out[f"prompt{p}.logits"].argmax(-1).tolist()
        print(f"prompt {p}: {len(ids)} tokens; next token after the prompt: {tok.decode([top1[-1]])!r}")
        if args.generate:
            print(f"  greedy: {tok.decode(ids + out[f'prompt{p}.gen_tokens'].tolist())!r}")
    print(f"smallest gap between the k-th and (k+1)-th choice score: {gap:.2e}")
    with open(args.out, "wb") as f:
        f.write(serialize(out))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes, {len(out)} tensors)")


if __name__ == "__main__":
    main()
