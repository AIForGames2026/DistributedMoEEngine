#!/usr/bin/env python3
"""TASK-32 reference for GLM-5.3-Flash (transformers' glm5_next): runs Hugging
Face's own Glm5NextForConditionalGeneration (text only, float32, eager
attention, CPU) on the weights in a llama.cpp GGUF file and writes golden
values like ref_qwen38.py, plus the pools DSA's indexer kept for every query
long enough to choose.
Testing only.

Loading undoes llama.cpp's converter (conversion/glm.py, Glm5NextModel):
A_log stored as -exp(A_log), dt_bias renamed, kv_b split into an absorbed
k_b (transposed) and v_b, and the KDA convolution kept as three tensors. The
routed experts stay quantized in the memory-mapped file and dequantize only
when a token picks them. With --lazy, the other matrices also stay quantized
and dequantize each time they are used, so the model fits in far less memory.

DSA scores a pool by a weighted sum of ReLUs, so pools can tie, and the model
does not say which tied pool to keep. Here ties go to the lower pool, as in
the engine, and the run reports how many choices had one.

Usage:
  python3 tools/ref/ref_glm53.py --gguf part1.gguf part2.gguf --config config.json \\
      --tokenizer tokenizer.json --out glm53_golden.tfref [--generate 32] [--lazy] \\
      [--long FILE --long-tokens 2600 --tail 32]
  python3 tools/ref/ref_glm53.py --tiny [--check]
      writes tests/golden/glm53_tiny.{gguf,tfref}: a tiny model with the
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
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS  # noqa: E402

ARCH = "glm5-next"


class LazyExperts(torch.nn.Module):
    """A layer's experts, each dequantized when a token picks it, with GLM's clamped SwiGLU."""

    def __init__(self, gate, up, down, limit):
        super().__init__()
        self.gate, self.up, self.down, self.limit = gate, up, down, limit

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
            gate = (x @ self.weight(self.gate, e).T).clamp(max=self.limit)
            up = (x @ self.weight(self.up, e).T).clamp(-self.limit, self.limit)
            y = (torch.nn.functional.silu(gate) * up) @ self.weight(self.down, e).T
            out.index_add_(0, tok, (y * top_k_weights[tok, pos, None]).to(out.dtype))
        return out


class LazyLinear(torch.nn.Module):
    """A bias-free linear layer whose weight stays quantized and dequantizes on each call."""

    def __init__(self, fetch):
        super().__init__()
        self.fetch = fetch
        self.register_buffer("weight", torch.empty(0), persistent=False)

    def forward(self, x):
        return x @ self.fetch().T


def hf_state(cfg, tensors):
    """transformers' state dict (without routed experts) from GGUF tensors given as
    name -> callable returning float32 numpy (outermost first)."""

    def t(name):
        return torch.from_numpy(np.ascontiguousarray(tensors[name]()))

    pre = "model.language_model."
    sd = {pre + "embed_tokens.weight": t("token_embd.weight"), pre + "norm.weight": t("output_norm.weight"),
          "lm_head.weight": t("output.weight")}
    heads, nope, vd = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
    for l, kind in enumerate(cfg.layer_types):
        g, h = f"blk.{l}.", f"{pre}layers.{l}."
        for hc, gg in (("attn_hc.", "hc_attn_"), ("ffn_hc.", "hc_ffn_")):
            for part in ("fn", "base", "scale"):
                sd[h + hc + part] = t(g + gg + part + ".weight")
        sd[h + "input_layernorm.weight"] = t(g + "attn_norm.weight")
        sd[h + "post_attention_layernorm.weight"] = t(g + "ffn_norm.weight")
        a = h + "self_attn."
        if kind == "linear_attention":
            for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output"),
                           ("b_proj", "ssm_beta"), ("g_a_proj", "ssm_g_a"), ("g_b_proj", "ssm_g_b"),
                           ("forget_gate.f_a_proj", "ssm_f_a"), ("forget_gate.f_b_proj", "ssm_f_b")):
                sd[a + hf + ".weight"] = t(g + gg + ".weight")
            sd[a + "conv1d.weight"] = torch.cat([t(g + f"ssm_conv1d_{x}.weight") for x in "qkv"])
            sd[a + "forget_gate.dt_bias"] = t(g + "ssm_dt.bias")
            sd[a + "forget_gate.A_log"] = torch.log(-t(g + "ssm_a"))
            sd[a + "o_norm.weight"] = t(g + "ssm_norm.weight")
        else:
            for hf, gg in (("q_a_proj", "attn_q_a"), ("q_b_proj", "attn_q_b"), ("kv_a_proj_with_mqa", "attn_kv_a_mqa"),
                           ("o_proj", "attn_output"), ("q_a_layernorm", "attn_q_a_norm"),
                           ("kv_a_layernorm", "attn_kv_a_norm")):
                sd[a + hf + ".weight"] = t(g + gg + ".weight")
            k_b = t(g + "attn_k_b.weight").transpose(1, 2)
            v_b = t(g + "attn_v_b.weight")
            sd[a + "kv_b_proj.weight"] = torch.cat([k_b, v_b], dim=1).reshape(heads * (nope + vd), -1)
            ix = a + "indexer."
            sd[ix + "wq_b.weight"] = t(g + "indexer.attn_q_b.weight")
            sd[ix + "wk.weight"] = t(g + "indexer.attn_k.weight")
            sd[ix + "k_norm.weight"] = t(g + "indexer.k_norm.weight")
            sd[ix + "k_norm.bias"] = t(g + "indexer.k_norm.bias")
            sd[ix + "weights_proj.weight"] = t(g + "indexer.proj.weight")
            sd[ix + "index_kpool_compress_ape"] = t(g + "indexer_compressor_ape.weight")
            sd[ix + "index_kpool_compress_gate"] = t(g + "indexer_compressor_gate.weight")
        m = h + "mlp."
        if cfg.mlp_layer_types[l] == "dense":
            for hf in ("gate", "up", "down"):
                sd[m + hf + "_proj.weight"] = t(g + f"ffn_{hf}.weight")
        else:
            sd[m + "gate.weight"] = t(g + "ffn_gate_inp.weight")
            sd[m + "gate.e_score_correction_bias"] = t(g + "exp_probs_b.bias")
            for hf in ("gate", "up", "down"):
                sd[m + "shared_experts." + hf + "_proj.weight"] = t(g + f"ffn_{hf}_shexp.weight")
    return sd


def load(cfg, tensors, experts, lazy=None):
    """The text model on these weights. `experts(layer)` gives (gate, up, down) for LazyExperts.
    `lazy`, when given, maps a state-dict key to a fetch function for weights kept quantized."""
    full = Glm5NextConfig(text_config=cfg.to_dict(), vision_config={"depth": 1, "hidden_size": 32, "num_heads": 2,
                                                                     "intermediate_size": 32, "out_hidden_size": 32})
    full.text_config.dtype = torch.float32
    full._attn_implementation = "eager"
    full.text_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Glm5NextForConditionalGeneration(full)
    model.model.visual = None
    sd = hf_state(cfg, tensors)
    lazy = lazy or {}
    for key in lazy:
        sd.pop(key, None)
    missing = model.load_state_dict(sd, strict=False, assign=True)
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k and k not in lazy and not k.startswith("model.visual")]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    for key, fetch in lazy.items():
        path = key.rsplit(".", 1)[0]
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent), name, LazyLinear(fetch))
    for l, layer in enumerate(model.model.language_model.layers):
        if cfg.mlp_layer_types[l] == "sparse":
            layer.mlp.experts = LazyExperts(*experts(l), cfg.swiglu_limit)
    meta = [n for n, x in list(model.named_parameters()) + list(model.named_buffers()) if x.is_meta]
    if meta:
        sys.exit(f"tensors left on the meta device: {meta[:5]}")
    return model.eval()


def quantize_q4_k(x):
    """Q4_K blocks for float32 rows whose length is a multiple of 256 (gguf-py only dequantizes
    Q4_K). Plain min/max rounding: the tests need exact bytes, not the best ones."""
    rows = x.reshape(-1, 256).astype(np.float32)
    sub = rows.reshape(-1, 8, 32)
    lo = np.minimum(sub.min(-1), 0.0)
    span = (sub.max(-1) - lo) / 15.0
    d = (span.max(-1) / 63.0).astype(np.float16)
    dmin = ((-lo).max(-1) / 63.0).astype(np.float16)
    df, mf = d.astype(np.float32)[:, None], dmin.astype(np.float32)[:, None]
    sc = np.clip(np.round(np.divide(span, df, out=np.zeros_like(span), where=df > 0)), 0, 63).astype(np.uint8)
    mn = np.clip(np.round(np.divide(-lo, mf, out=np.zeros_like(lo), where=mf > 0)), 0, 63).astype(np.uint8)
    step = (df * sc)[..., None]
    q = np.divide(sub + (mf * mn)[..., None], step, out=np.zeros_like(sub), where=step > 0)
    q = np.clip(np.round(q), 0, 15).astype(np.uint8).reshape(-1, 256)
    scales = np.zeros((rows.shape[0], 12), np.uint8)
    scales[:, 0:4] = (sc[:, 0:4] & 63) | ((sc[:, 4:8] >> 4) << 6)
    scales[:, 4:8] = (mn[:, 0:4] & 63) | ((mn[:, 4:8] >> 4) << 6)
    scales[:, 8:12] = (sc[:, 4:8] & 15) | ((mn[:, 4:8] & 15) << 4)
    qs = np.zeros((rows.shape[0], 128), np.uint8)
    for c in range(4):
        qs[:, c * 32:(c + 1) * 32] = q[:, c * 64:c * 64 + 32] | (q[:, c * 64 + 32:c * 64 + 64] << 4)
    blocks = np.concatenate([d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2), scales, qs], axis=1)
    return blocks.reshape(*x.shape[:-1], x.shape[-1] // 256 * 144)


def dequant(t):
    shape = tuple(int(d) for d in reversed(t.shape))
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(t.data, dtype=np.float32).reshape(shape)
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(shape).astype(np.float32)


def text_config(path):
    with open(path) as f:
        c = json.load(f)
    full = Glm5NextConfig(**c)
    cfg = full.text_config
    cfg.dtype = torch.float32
    return cfg


def build(gguf_paths, config_path, lazy):
    cfg = text_config(config_path)
    readers = [gguf.GGUFReader(p) for p in gguf_paths]
    by_name = {t.name: t for r in readers for t in r.tensors}
    small = {n: (lambda t=t: dequant(t)) for n, t in by_name.items() if "_exps." not in n}
    keep = {}
    if lazy:
        # Matrices worth keeping quantized: every Q8_0 or BF16 weight with more than a million values.
        names = {k: v for k, v in hf_state_names(cfg).items() if v in by_name}
        for key, gname in names.items():
            t = by_name[gname]
            if t.tensor_type != gguf.GGMLQuantizationType.F32 and int(np.prod(t.shape)) > (1 << 20) and key.endswith("proj.weight"):
                keep[key] = (lambda t=t: torch.from_numpy(dequant(t)))
    experts = lambda l: tuple(by_name[f"blk.{l}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))  # noqa: E731
    if keep:
        lazy_g = set(names[k] for k in keep)
        small = {n: f for n, f in small.items() if n not in lazy_g}
        small.update({n: (lambda: np.zeros((1, 1), np.float32)) for n in lazy_g})
    return load(cfg, small, experts, keep), cfg


def hf_state_names(cfg):
    """The state-dict keys of plain linear layers mapped to their GGUF tensor names."""
    pre = "model.language_model."
    out = {"lm_head.weight": "output.weight"}
    for l, kind in enumerate(cfg.layer_types):
        g, h = f"blk.{l}.", f"{pre}layers.{l}."
        a = h + "self_attn."
        if kind == "linear_attention":
            for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                out[a + hf + ".weight"] = g + gg + ".weight"
        else:
            for hf, gg in (("q_a_proj", "attn_q_a"), ("q_b_proj", "attn_q_b"), ("o_proj", "attn_output")):
                out[a + hf + ".weight"] = g + gg + ".weight"
        m = h + "mlp."
        if cfg.mlp_layer_types[l] == "dense":
            for hf in ("gate", "up", "down"):
                out[m + hf + "_proj.weight"] = g + f"ffn_{hf}.weight"
        else:
            for hf in ("gate", "up", "down"):
                out[m + "shared_experts." + hf + "_proj.weight"] = g + f"ffn_{hf}_shexp.weight"
    return out


def dsa_layers(cfg):
    return [l for l, kind in enumerate(cfg.layer_types) if kind != "linear_attention"]


class LowerPoolTies:
    """While on, a topk over the last dimension (the indexer's choice of pools) keeps the lower
    index among equal scores, and counts the rows where a tie fell at the cut."""

    def __init__(self):
        self.on = False
        self.ties = 0
        self.choices = 0
        self.orig = torch.Tensor.topk
        outer = self

        def topk(t, k, dim=-1, largest=True, sorted=True):
            if not outer.on or dim not in (-1, t.dim() - 1) or not largest:
                return outer.orig(t, k, dim, largest, sorted)
            vals, idx = torch.sort(t, dim=-1, descending=True, stable=True)
            rows = vals.reshape(-1, vals.shape[-1])
            outer.choices += rows.shape[0]
            if k < t.shape[-1]:
                outer.ties += int((rows[:, k - 1] == rows[:, k]).sum())
            return torch.return_types.topk((vals[..., :k], idx[..., :k]))

        torch.Tensor.topk = topk

    def hooks(self, model, cfg):
        def start(*_):
            self.on = True

        def stop(*_):
            self.on = False

        out = []
        for l in dsa_layers(cfg):
            ix = model.model.language_model.layers[l].self_attn.indexer
            out.append(ix.register_forward_pre_hook(start))
            out.append(ix.register_forward_hook(stop))
        return out


TIES = LowerPoolTies()


def pools_kept(topk, cfg):
    """From one indexer's output [queries, width] of token indices, the sorted complete pools each
    query kept, for the queries with more complete pools than the budget (the others keep them all)."""
    pool = cfg.index_kpool
    top = cfg.index_topk // pool
    q = topk.shape[0]
    first = next((i for i in range(q) if (i + 1) // pool > top), None)
    if first is None:
        return None, None
    out = torch.empty(q - first, top, dtype=torch.int32)
    for i in range(first, q):
        heads = topk[i, :top * pool:pool]
        if (heads < 0).any():
            sys.exit(f"query {i} kept fewer than {top} pools")
        out[i - first] = torch.sort(heads // pool).values.to(torch.int32)
    return first, out


def choice_gap(logits, bias, k):
    """The smallest gap, over rows, between the k-th and (k+1)-th choice score (sigmoid plus bias)."""
    s = torch.topk(torch.sigmoid(logits.float()) + bias, k + 1, dim=-1).values
    return float((s[..., k - 1] - s[..., k]).min())


def record(model, cfg, prompts, gen, tail=None):
    """Golden values for each token-id list in `prompts`, plus `gen` tokens of greedy
    generate(). `tail[p]`, when set, keeps only that many final positions' logits
    and no per-layer hidden states for prompt p."""
    layers = model.model.language_model.layers
    sparse = [l for l in range(cfg.num_hidden_layers) if cfg.mlp_layer_types[l] == "sparse"]
    out = {"config": torch.tensor([len(prompts), cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok,
                                   cfg.hidden_size, cfg.vocab_size], dtype=torch.int32)}
    k = cfg.num_experts_per_tok
    min_gap = float("inf")
    seen = {}

    def on_gate(l):
        def hook(mod, _args, res):
            if seen.get("on"):
                seen[("gate", l)] = (res[0].clone(), res[1].clone(), res[2].clone(), mod.e_score_correction_bias)
        return hook

    def on_layer(l):
        def hook(_mod, _args, res):
            if seen.get("on") and seen.get("hidden"):
                seen[("hidden", l)] = res[0][0].clone()
        return hook

    def on_indexer(l):
        def hook(_mod, _args, res):
            if seen.get("on"):
                seen[("dsa", l)] = res[0].clone()
        return hook

    hooks = TIES.hooks(model, cfg)
    for l in sparse:
        hooks.append(layers[l].mlp.gate.register_forward_hook(on_gate(l)))
    for l in range(cfg.num_hidden_layers):
        hooks.append(layers[l].register_forward_hook(on_layer(l)))
    for l in dsa_layers(cfg):
        hooks.append(layers[l].self_attn.indexer.register_forward_hook(on_indexer(l)))
    with torch.no_grad():
        for p, ids in enumerate(prompts):
            short = tail is None or tail[p] is None
            seen.clear()
            seen["on"], seen["hidden"] = True, short
            res = model(torch.tensor([ids]), use_cache=False)
            seen["on"] = False
            out[f"prompt{p}.tokens"] = torch.tensor(ids, dtype=torch.int32)
            if short:
                out[f"prompt{p}.logits"] = res.logits[0].contiguous()
            else:
                n = min(tail[p], len(ids))
                out[f"prompt{p}.logits"] = res.logits[0, -n:].contiguous()
                out[f"prompt{p}.logits_from"] = torch.tensor([len(ids) - n], dtype=torch.int32)
            for l in range(cfg.num_hidden_layers):
                if short:
                    out[f"prompt{p}.layer{l}.hidden"] = seen[("hidden", l)].contiguous()
                if l in sparse:
                    logits, w, idx, bias = seen[("gate", l)]
                    min_gap = min(min_gap, choice_gap(logits, bias, k))
                    order = torch.argsort(idx, dim=-1)
                    out[f"prompt{p}.layer{l}.topk_idx"] = torch.gather(idx, -1, order).to(torch.int32).contiguous()
                    out[f"prompt{p}.layer{l}.topk_w"] = torch.gather(w, -1, order).contiguous()
                if ("dsa", l) in seen:
                    first, kept = pools_kept(seen[("dsa", l)], cfg)
                    if first is not None:
                        out[f"prompt{p}.layer{l}.dsa_from"] = torch.tensor([first], dtype=torch.int32)
                        out[f"prompt{p}.layer{l}.dsa_pools"] = kept.contiguous()
            if gen:
                # Each step's smallest gap between the k-th and (k+1)-th choice score, over every
                # MoE layer at the position its logits come from; a near-tie there can flip a choice.
                gaps = []

                def gap(mod, _args, res):
                    g_now = choice_gap(res[0][-1:], mod.e_score_correction_bias, k)
                    if mod is layers[sparse[0]].mlp.gate:
                        gaps.append(g_now)
                    else:
                        gaps[-1] = min(gaps[-1], g_now)

                gate_hooks = [layers[l].mlp.gate.register_forward_hook(gap) for l in sparse]
                # No end-of-text stop, so every prompt yields exactly `gen` tokens.
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


TINY = dict(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=5,
            num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
            num_experts_per_tok=4, routed_scaling_factor=2.5, kv_lora_rank=32, q_lora_rank=64, qk_rope_head_dim=0,
            qk_nope_head_dim=32, v_head_dim=32, n_group=1, topk_group=1, norm_topk_prob=True, rms_norm_eps=1e-5,
            max_position_embeddings=128, tie_word_embeddings=False, hidden_act="silu",
            mlp_layer_types=["dense"] + ["sparse"] * 4, index_topk=8, index_head_dim=32, index_n_heads=2,
            indexer_types=["full"] * 5, swiglu_limit=1.0, linear_head_dim=16, linear_num_heads=4,
            linear_conv_kernel_dim=4, linear_lower_bound=-5.0, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20,
            index_kpool=4, index_kpool_always_select_tail=True, pad_token_id=0, eos_token_id=2)
TINY_SEED = 20261117
# Both long prompts have more complete pools than DSA's budget of two.
TINY_PROMPTS = [[3, 17, 200, 45, 9],
                [11, 12, 13, 14, 15, 16, 17, 18, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [250, 128, 64, 32, 16, 8, 4, 99, 98, 97, 96, 95, 94, 93]]
TINY_GENERATE = 16
SETS = (("glm53_tiny", TINY, TINY_PROMPTS),)
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir, stem="glm53_tiny", config=None, prompts=None):
    """A tiny GLM-5.3-Flash (three KDA layers, an MLA layer, then KDA again; the first
    layer dense) with random weights, in llama.cpp's layout and weight types:
    Q8_0 matrices and experts, BF16 KDA gates and MLA latents, F32 for the rest.
    Q4_K needs rows of 256, so it is checked on a separate tensor, `test.q4_k`,
    whose dequantized values go in the golden file. The reference runs on the
    dequantized values."""
    from transformers import Glm5NextTextConfig

    cfg = Glm5NextTextConfig(**(config or TINY))
    cfg.dtype = torch.float32
    prompts = prompts or TINY_PROMPTS
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8, Q4K, BF16 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.BF16
    d, f, e, v, n = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts, cfg.vocab_size, cfg.num_hidden_layers
    heads, lhd = cfg.num_attention_heads, cfg.linear_head_dim
    qkv = heads * lhd
    hc = cfg.hc_mult
    nope, vd, kvr, qr = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank, cfg.q_lora_rank
    ih, inh = cfg.index_head_dim, cfg.index_n_heads

    w = gguf.GGUFWriter(path=None, arch=ARCH)
    key = lambda k: f"{ARCH}.{k}"  # noqa: E731
    u32 = lambda k, x: w.add_uint32(key(k), x)  # noqa: E731
    f32v = lambda k, x: w.add_float32(key(k), x)  # noqa: E731
    w.add_block_count(n)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(d)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(heads)
    w.add_array(key("attention.head_count_kv"), [0 if k == "linear_attention" else 1 for k in cfg.layer_types])
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    u32("expert_count", e)
    u32("expert_used_count", cfg.num_experts_per_tok)
    u32("expert_group_count", cfg.n_group)
    u32("expert_group_used_count", cfg.topk_group)
    u32("expert_gating_func", 2)
    u32("vocab_size", v)
    f32v("attention.layer_norm_epsilon", 1e-6)
    u32("ssm.conv_kernel", cfg.linear_conv_kernel_dim)
    u32("kda.head_dim", lhd)
    f32v("kda.gate_lower_bound", cfg.linear_lower_bound)
    u32("attention.q_lora_rank", qr)
    u32("attention.kv_lora_rank", kvr)
    u32("rope.dimension_count", 0)
    u32("attention.key_length", kvr)
    u32("attention.value_length", kvr)
    u32("attention.key_length_mla", nope)
    u32("attention.value_length_mla", vd)
    u32("attention.indexer.head_count", inh)
    u32("attention.indexer.key_length", ih)
    u32("attention.indexer.top_k", cfg.index_topk)
    u32("attention.indexer.kpool", cfg.index_kpool)
    w.add_bool(key("attention.indexer.kpool_select_tail"), cfg.index_kpool_always_select_tail)
    w.add_array(key("attention.indexer.types"), [t == "full" for t in cfg.indexer_types])
    u32("hyper_connection.count", hc)
    u32("hyper_connection.sinkhorn_iterations", cfg.hc_sinkhorn_iters)
    f32v("hyper_connection.epsilon", cfg.hc_eps)
    u32("leading_dense_block_count", cfg.mlp_layer_types.count("dense"))
    u32("expert_feed_forward_length", f)
    u32("expert_shared_count", cfg.n_shared_experts)
    f32v("expert_weights_scale", cfg.routed_scaling_factor)
    w.add_bool(key("expert_weights_norm"), cfg.norm_topk_prob)
    w.add_array(key("swiglu_clamp_exp"), [float(cfg.swiglu_limit)] * n)
    w.add_array(key("swiglu_clamp_shexp"), [float(cfg.swiglu_limit)] * n)

    tensors = {}
    stacked = {}

    def keep(name, x):
        tensors[name] = (lambda x=x: x)

    def quant(name, x, kind):
        q = quantize_q4_k(x.numpy()) if kind == Q4K else gguf.quants.quantize(x.numpy(), kind)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=kind)
        return gguf.quants.dequantize(q, kind).reshape(x.shape).astype(np.float32)

    def linear(name, shape, scale=1.0, kind=Q8):
        keep(name, quant(name, torch.randn(*shape, generator=g) * scale / shape[-1] ** 0.5, kind))

    def f32(name, x):
        x = x.numpy().astype(np.float32)
        w.add_tensor(name, x)
        keep(name, x)

    def gamma(name, size):
        f32(name, 1.0 + torch.randn(size, generator=g) * 0.1)

    keep("token_embd.weight", quant("token_embd.weight", torch.randn(v, d, generator=g), Q8))
    for l, kind in enumerate(cfg.layer_types):
        p = f"blk.{l}."
        for part in ("hc_attn_", "hc_ffn_"):
            f32(p + part + "fn.weight", torch.randn((2 + hc) * hc, hc * d, generator=g) / (hc * d) ** 0.5)
            f32(p + part + "base.weight", torch.randn((2 + hc) * hc, generator=g) * 0.5)
            f32(p + part + "scale.weight", 0.5 + torch.rand(3, generator=g))
        gamma(p + "attn_norm.weight", d)
        gamma(p + "ffn_norm.weight", d)
        if kind == "linear_attention":
            for name in ("attn_q", "attn_k", "attn_v"):
                linear(p + name + ".weight", (qkv, d))
            linear(p + "attn_output.weight", (d, qkv))
            for x in "qkv":
                f32(p + f"ssm_conv1d_{x}.weight", torch.randn(qkv, 1, cfg.linear_conv_kernel_dim, generator=g) * 0.5)
            linear(p + "ssm_beta.weight", (heads, d), kind=BF16)
            linear(p + "ssm_f_a.weight", (lhd, d), kind=BF16)
            linear(p + "ssm_f_b.weight", (qkv, lhd), kind=BF16)
            linear(p + "ssm_g_a.weight", (lhd, d), kind=BF16)
            linear(p + "ssm_g_b.weight", (qkv, lhd), kind=BF16)
            f32(p + "ssm_a", -torch.empty(heads).uniform_(0.5, 4, generator=g))
            f32(p + "ssm_dt.bias", torch.randn(qkv, generator=g) * 0.5)
            gamma(p + "ssm_norm.weight", lhd)
        else:
            linear(p + "attn_q_a.weight", (qr, d))
            gamma(p + "attn_q_a_norm.weight", qr)
            linear(p + "attn_q_b.weight", (heads * nope, qr))
            linear(p + "attn_kv_a_mqa.weight", (kvr, d), kind=BF16)
            gamma(p + "attn_kv_a_norm.weight", kvr)
            linear(p + "attn_k_b.weight", (heads, kvr, nope), kind=BF16)
            linear(p + "attn_v_b.weight", (heads, vd, kvr), kind=BF16)
            linear(p + "attn_output.weight", (d, heads * vd))
            linear(p + "indexer.attn_q_b.weight", (inh * ih, qr), kind=BF16)
            linear(p + "indexer.attn_k.weight", (ih, d), kind=BF16)
            gamma(p + "indexer.k_norm.weight", ih)
            f32(p + "indexer.k_norm.bias", torch.randn(ih, generator=g) * 0.1)
            f32(p + "indexer.proj.weight", torch.randn(inh, d, generator=g) / d ** 0.5)
            f32(p + "indexer_compressor_ape.weight", torch.randn(cfg.index_kpool, ih, generator=g))
            f32(p + "indexer_compressor_gate.weight", torch.randn(ih, d, generator=g) / d ** 0.5)
        if cfg.mlp_layer_types[l] == "dense":
            linear(p + "ffn_gate.weight", (cfg.intermediate_size, d))
            linear(p + "ffn_up.weight", (cfg.intermediate_size, d))
            linear(p + "ffn_down.weight", (d, cfg.intermediate_size))
        else:
            f32(p + "ffn_gate_inp.weight", torch.randn(e, d, generator=g) * (4.0 / d ** 0.5))
            f32(p + "exp_probs_b.bias", torch.randn(e, generator=g) * 0.05)
            linear(p + "ffn_gate_shexp.weight", (f, d))
            linear(p + "ffn_up_shexp.weight", (f, d))
            linear(p + "ffn_down_shexp.weight", (d, f))
            for gg, shape in (("gate", (f, d)), ("up", (f, d)), ("down", (d, f))):
                stacked[(l, gg)] = torch.from_numpy(quant(p + f"ffn_{gg}_exps.weight",
                                                          torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5, Q8))
    gamma("output_norm.weight", d)
    linear("output.weight", (v, d))
    q4k = quant("test.q4_k", torch.randn(3, 512, generator=g) * 2.0, Q4K)

    experts = lambda l: tuple((lambda x, s=stacked[(l, p)]: s[x]) for p in ("gate", "up", "down"))  # noqa: E731
    model = load(cfg, tensors, experts)
    path = os.path.join(out_dir, stem + ".gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, prompts, TINY_GENERATE)
    out["test.q4_k"] = torch.from_numpy(q4k)
    if not any(k.endswith("dsa_pools") for k in out):
        sys.exit("no tiny prompt is long enough for DSA to choose pools")
    with open(os.path.join(out_dir, stem + ".tfref"), "wb") as fh:
        fh.write(serialize(out))
    return gap


def tiny_main(check):
    import hashlib
    import tempfile

    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    for stem, config, prompts in SETS:
        names = (stem + ".gguf", stem + ".tfref")
        if not check:
            gap = write_tiny(GOLDEN_DIR, stem, config, prompts)
            for name in names:
                print(f"wrote tests/golden/{name} ({os.path.getsize(os.path.join(GOLDEN_DIR, name))} bytes)")
            print(f"smallest gap between the k-th and (k+1)-th choice score: {gap:.2e}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny(tmp, stem, config, prompts)
            bad = [name for name in names if digest(os.path.join(tmp, name)) != digest(os.path.join(GOLDEN_DIR, name))]
        if bad:
            sys.exit(f"MISMATCH: {', '.join(bad)} differ from a fresh run")
        print(f"OK: {names[0]} and {names[1]} match a fresh run")
    print(f"DSA pool choices with a tie at the cut: {TIES.ties} of {TIES.choices} (kept the lower pool)")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, nargs="+", help="the GGUF file, or each part of a split one")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0, help="also record this many tokens of greedy generate()")
    ap.add_argument("--lazy", action="store_true", help="keep large matrices quantized, dequantizing on each use")
    ap.add_argument("--long", help="also run a prompt made of this file's first --long-tokens tokens")
    ap.add_argument("--long-tokens", type=int, default=2600)
    ap.add_argument("--tail", type=int, default=32, help="logits kept for the long prompt's last positions")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    model, cfg = build(args.gguf, args.config, args.lazy)
    tok = Tokenizer.from_file(args.tokenizer)
    prompts = [tok.encode(text).ids for text in PROMPTS]
    tail = [None] * len(prompts)
    if args.long:
        with open(args.long, encoding="utf-8") as f:
            ids = tok.encode(f.read()).ids[:args.long_tokens]
        if len(ids) < args.long_tokens:
            sys.exit(f"{args.long} has only {len(ids)} tokens")
        prompts.append(ids)
        tail.append(args.tail)
    out, gap = record(model, cfg, prompts, args.generate, tail)
    for p, ids in enumerate(prompts):
        top1 = out[f"prompt{p}.logits"].argmax(-1).tolist()
        print(f"prompt {p}: {len(ids)} tokens; next token after the prompt: {tok.decode([top1[-1]])!r}")
        if args.generate:
            print(f"  greedy: {tok.decode(out[f'prompt{p}.gen_tokens'].tolist())!r}")
    print(f"smallest gap between the k-th and (k+1)-th choice score: {gap:.2e}")
    print(f"DSA pool choices with a tie at the cut: {TIES.ties} of {TIES.choices} (kept the lower pool)")
    with open(args.out, "wb") as f:
        f.write(serialize(out))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes, {len(out)} tensors)")


if __name__ == "__main__":
    main()
