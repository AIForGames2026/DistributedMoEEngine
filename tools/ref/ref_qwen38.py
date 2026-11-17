#!/usr/bin/env python3
"""TASK-31 reference for Qwen3.8-Flash-Next (transformers' qwen4_exp): runs
Hugging Face's own Qwen4ExpForCausalLM (float32, eager attention, CPU) on the
weights in a llama.cpp GGUF file and writes golden values like ref_qwen35.py,
plus the blocks QSA's indexer kept for every query long enough to choose.
Testing only.

Loading undoes llama.cpp's converter (conversion/qwen4exp.py): RMSNorm gammas
stored plus one (all but the Gated DeltaNet's gated norm), A_log as -exp(A_log),
value heads in tiled order, and pairs of weights that read the same input
merged as [a; b] (qkv with z, beta with alpha, each hyper-connection's down
with its inject). The experts and the n-gram table stay quantized in the
memory-mapped file and dequantize only the rows a prompt uses, since the
whole model in float32 would not fit in memory.

A long prompt keeps routing and QSA choices for every position but logits
only for its last `--tail` positions and no per-layer hidden states, which
would otherwise take gigabytes.

QSA scores a block by a sum of ReLUs, so blocks can tie (at zero, say), and
the model does not say which tied block to keep. transformers leaves it to
torch.topk, whose order among ties is not specified. Here ties go to the
lower block, as in the engine, and the run reports how many choices had one.

Usage:
  python3 tools/ref/ref_qwen38.py --gguf part1.gguf part2.gguf --config config.json \\
      --tokenizer tokenizer.json --out qwen38_golden.tfref [--generate 32] \\
      [--long FILE --long-tokens 2600 --tail 32]
  python3 tools/ref/ref_qwen38.py --tiny [--check]
      writes tests/golden/qwen38_tiny.{gguf,tfref}: a tiny model with the
      same layer mix and random weights (seeded), and transformers' outputs;
      and qwen38_gpu.{gguf,tfref}, the same with 256-wide attention heads
      for checking the GPU kernels.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import gguf
from tokenizers import Tokenizer
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_moe import serialize  # noqa: E402
from ref_olmoe import PROMPTS  # noqa: E402
from ref_qwen35 import LazyExperts, tiled_to_grouped  # noqa: E402

ARCH = "qwen4exp"


class LazyRows(torch.nn.Module):
    """The n-gram table: only the rows a batch of ids names are dequantized."""

    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.register_buffer("weight", torch.empty(0), persistent=False)

    def forward(self, ids):
        uniq, inv = torch.unique(ids, return_inverse=True)
        table = torch.from_numpy(np.ascontiguousarray(self.rows(uniq.numpy()), dtype=np.float32))
        return table[inv]


def hf_state(cfg, tensors):
    """transformers' state dict (without experts and the n-gram table) from GGUF tensors
    given as name -> float32 numpy (outermost first)."""
    k_heads, v_heads = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    r, hv = v_heads // k_heads, cfg.linear_value_head_dim
    key_dim = cfg.linear_key_head_dim * k_heads
    conv_dim = 2 * key_dim + v_heads * hv
    low = cfg.hc_lowrank

    def t(name):
        return torch.from_numpy(np.ascontiguousarray(tensors[name]))

    def norm(name):
        return t(name) - 1.0

    def untile(x, dim, head_dim):
        return tiled_to_grouped(x, dim, k_heads, r, head_dim)

    def mixer(prefix, gg, combine=True):
        sd[prefix + "hc_norm.weight"] = norm(gg + "_norm.weight")
        sd[prefix + "input_mix_weight_up.weight"] = t(gg + "_up.weight")
        if combine:
            di = t(gg + "_down_inject.weight")
            sd[prefix + "input_mix_weight_down.weight"] = di[:low]
            sd[prefix + "block_inject_weight.weight"] = di[low:]
        else:
            sd[prefix + "input_mix_weight_down.weight"] = t(gg + "_down.weight")

    sd = {"model.embed_tokens.weight": t("token_embd.weight"), "lm_head.weight": t("output.weight")}
    mixer("model.hyper_connection_mixer.", "output_hc", combine=False)
    for l, kind in enumerate(cfg.layer_types):
        g, h = f"blk.{l}.", f"model.layers.{l}."
        mixer(h + "attn_hyper_connection.", g + "hc_attn")
        mixer(h + "mlp_hyper_connection.", g + "hc_ffn")
        if kind == "linear_attention":
            a = h + "linear_attn."
            qkvz = t(g + "attn_qkv_gate.weight")
            qkv, z = qkvz[:conv_dim], qkvz[conv_dim:]
            sd[a + "in_proj_qkv.weight"] = torch.cat([qkv[:2 * key_dim], untile(qkv[2 * key_dim:], 0, hv)])
            sd[a + "in_proj_z.weight"] = untile(z, 0, hv)
            ba = t(g + "ssm_beta_alpha.weight")
            sd[a + "in_proj_b.weight"] = untile(ba[:v_heads], 0, 1)
            sd[a + "in_proj_a.weight"] = untile(ba[v_heads:], 0, 1)
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
            sd[a + "indexer.index_qk_proj.weight"] = torch.cat([t(g + "indexer.q_proj.weight"),
                                                                t(g + "indexer.k_proj.weight")])
            sd[a + "indexer.q_layernorm.weight"] = norm(g + "indexer.q_norm.weight")
            sd[a + "indexer.k_layernorm.weight"] = norm(g + "indexer.k_norm.weight")
        if l + 1 in cfg.ple_layer_ids:
            p = h + "ple."
            sd[p + "key_proj.weight"] = t(g + "ple_key.weight")
            sd[p + "value_proj.weight"] = t(g + "ple_value.weight")
            for n in ("key", "query", "conv"):
                sd[p + f"norm_{n}.weight"] = norm(g + f"ple_norm_{n}.weight")
            sd[p + "conv1d.weight"] = t(g + "ple_conv1d.weight").unsqueeze(1)
        m = h + "mlp."
        sd[m + "gate.weight"] = t(g + "ffn_gate_inp.weight")
        for hf, gg in (("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")):
            sd[m + "shared_expert." + hf + ".weight"] = t(g + gg + ".weight")
        sd[m + "shared_expert_gate.weight"] = t(g + "ffn_gate_inp_shexp.weight").reshape(1, -1)
    return sd


def load(cfg, tensors, experts, ngram_rows, hash_constants):
    """The model on these weights. `experts(layer)` gives (gate, up, down) for LazyExperts,
    `ngram_rows(ids)` the n-gram table's rows, and `hash_constants` the GGUF's
    (multipliers, offsets, vocab sizes), checked against transformers' own."""
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)
    missing = model.load_state_dict(hf_state(cfg, tensors), strict=False, assign=True)
    left = [k for k in missing.missing_keys if ".mlp.experts." not in k and ".ple_embedding." not in k]
    if left or missing.unexpected_keys:
        sys.exit(f"weights not mapped: missing {left[:5]}, unexpected {missing.unexpected_keys[:5]}")
    # Buffers built on the meta device hold no values; rebuild them on the CPU.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    for l, layer in enumerate(model.model.layers):
        layer.mlp.experts = LazyExperts(*experts(l))
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            want = type(emb)(cfg, cfg.ple_embed_dim, l, emb.ple_layer_index) if cfg.vocab_size < 4096 else None
            mult, offs, sizes = (torch.tensor(x, dtype=torch.long) for x in hash_constants)
            if want is not None and not (torch.equal(want.layer_multipliers, mult)
                                         and torch.equal(want.ngram_heads_offsets, offs)
                                         and torch.equal(want.ngram_heads_vocab_sizes, sizes)):
                sys.exit("the GGUF's n-gram hash constants differ from transformers'")
            emb.layer_multipliers, emb.ngram_heads_offsets, emb.ngram_heads_vocab_sizes = mult, offs, sizes
            emb.ngram_embedding = LazyRows(ngram_rows)
    meta = [n for n, x in list(model.named_parameters()) + list(model.named_buffers()) if x.is_meta]
    if meta:
        sys.exit(f"tensors left on the meta device: {meta[:5]}")
    return model.eval()


def dequant(t):
    shape = tuple(int(d) for d in reversed(t.shape))
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(t.data, dtype=np.float32).reshape(shape)
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(shape).astype(np.float32)


def field(reader, key):
    f = reader.fields[key]
    return [int(f.parts[i][0]) for i in f.data]


def text_config(path):
    with open(path) as f:
        c = json.load(f)
    cfg = Qwen4ExpTextConfig(**c.get("text_config", c))
    cfg.dtype = torch.float32
    return cfg


def build(gguf_paths, config_path):
    cfg = text_config(config_path)
    readers = [gguf.GGUFReader(p) for p in gguf_paths]
    by_name = {t.name: t for r in readers for t in r.tensors}
    tensors = {n: dequant(t) for n, t in by_name.items() if "_exps." not in n and n != "per_layer_token_embd.weight"}
    experts = lambda l: tuple(by_name[f"blk.{l}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))  # noqa: E731
    table = by_name["per_layer_token_embd.weight"]
    rows = lambda ids: gguf.quants.dequantize(np.asarray(table.data[ids]), table.tensor_type)  # noqa: E731
    consts = tuple(field(readers[0], f"{ARCH}.ple.{k}") for k in ("layer_multipliers", "head_offsets", "head_vocab_sizes"))
    return load(cfg, tensors, experts, rows, consts), cfg


def qsa_layers(cfg):
    return [l for l, kind in enumerate(cfg.layer_types) if kind != "linear_attention"]


class LowerBlockTies:
    """While on, a 1-D topk (the indexer's choice of blocks) keeps the lower index among equal
    scores, and counts the choices where a tie fell at the cut."""

    def __init__(self):
        self.on = False
        self.ties = 0
        self.choices = 0
        self.orig = torch.Tensor.topk
        outer = self

        def topk(t, k, dim=-1, largest=True, sorted=True):
            if not outer.on or t.dim() != 1 or not largest:
                return outer.orig(t, k, dim, largest, sorted)
            vals, idx = torch.sort(t, descending=True, stable=True)
            outer.choices += 1
            if k < t.numel() and vals[k - 1] == vals[k]:
                outer.ties += 1
            return torch.return_types.topk((vals[:k], idx[:k]))

        torch.Tensor.topk = topk

    def hooks(self, model, cfg):
        def start(*_):
            self.on = True

        def stop(*_):
            self.on = False

        out = []
        for l in qsa_layers(cfg):
            ix = model.model.layers[l].self_attn.indexer
            out.append(ix.register_forward_pre_hook(start))
            out.append(ix.register_forward_hook(stop))
        return out


TIES = LowerBlockTies()


def blocks_kept(mask, cfg):
    """From one indexer's token mask [queries, keys], the sorted complete blocks each query kept,
    for the queries with more complete blocks than the budget (the others keep them all)."""
    ratio = cfg.indexer_compress_ratio
    top = cfg.indexer_budget // ratio
    q = mask.shape[0]
    first = next((i for i in range(q) if (i + 1) // ratio > top), None)
    if first is None:
        return None, None
    out = torch.empty(q - first, top, dtype=torch.int32)
    for i in range(first, q):
        full = (i + 1) // ratio
        picked = mask[i, :full * ratio].reshape(full, ratio)[:, 0].nonzero().flatten()
        if picked.numel() != top:
            sys.exit(f"query {i} kept {picked.numel()} blocks, not {top}")
        out[i - first] = picked.to(torch.int32)
    return first, out


def record(model, cfg, prompts, gen, tail=None):
    """Golden values for each token-id list in `prompts`, plus `gen` tokens of greedy
    generate(). `tail[p]`, when set, keeps only that many final positions' logits
    and no per-layer hidden states for prompt p."""
    out = {"config": torch.tensor([len(prompts), cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok,
                                   cfg.hidden_size, cfg.vocab_size], dtype=torch.int32)}
    k = cfg.num_experts_per_tok
    min_gap = float("inf")
    masks = {}
    hooks = TIES.hooks(model, cfg)
    for l in qsa_layers(cfg):
        def keep(_mod, _args, res, l=l):
            if masks.get("on"):
                masks[l] = res[0, 0].clone()
        hooks.append(model.model.layers[l].self_attn.indexer.register_forward_hook(keep))
    with torch.no_grad():
        for p, ids in enumerate(prompts):
            short = tail is None or tail[p] is None
            masks.clear()
            masks["on"] = True
            res = model(torch.tensor([ids]), output_hidden_states=short, output_router_logits=True, use_cache=False)
            masks["on"] = False
            out[f"prompt{p}.tokens"] = torch.tensor(ids, dtype=torch.int32)
            if short:
                out[f"prompt{p}.logits"] = res.logits[0].contiguous()
            else:
                n = min(tail[p], len(ids))
                out[f"prompt{p}.logits"] = res.logits[0, -n:].contiguous()
                out[f"prompt{p}.logits_from"] = torch.tensor([len(ids) - n], dtype=torch.int32)
            for l in range(cfg.num_hidden_layers):
                # hidden_states[l + 1] is layer l's output (all streams), except the last.
                if short and l + 1 < cfg.num_hidden_layers:
                    out[f"prompt{p}.layer{l}.hidden"] = res.hidden_states[l + 1][0].contiguous()
                probs = torch.softmax(res.router_logits[l].float(), dim=-1)
                w, idx = torch.topk(probs, k + 1, dim=-1)
                min_gap = min(min_gap, float((w[:, k - 1] - w[:, k]).min()))
                out[f"prompt{p}.layer{l}.topk_idx"] = idx[:, :k].to(torch.int32).contiguous()
                out[f"prompt{p}.layer{l}.topk_w"] = w[:, :k].contiguous()
                if l in masks:
                    first, kept = blocks_kept(masks[l], cfg)
                    if first is not None:
                        out[f"prompt{p}.layer{l}.qsa_from"] = torch.tensor([first], dtype=torch.int32)
                        out[f"prompt{p}.layer{l}.qsa_blocks"] = kept.contiguous()
            if gen:
                # Each step's smallest gap between the k-th and (k+1)-th router probability, over every
                # layer at the position its logits come from; a near-tie there can flip a choice.
                gaps = []

                def gap(mod, _args, res):
                    probs = torch.softmax(res[0][-1].float(), dim=-1)
                    w = torch.topk(probs, k + 1).values
                    g_now = float(w[k - 1] - w[k])
                    if mod is model.model.layers[0].mlp.gate:
                        gaps.append(g_now)
                    else:
                        gaps[-1] = min(gaps[-1], g_now)

                gate_hooks = [layer.mlp.gate.register_forward_hook(gap) for layer in model.model.layers]
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


TINY = dict(vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16,
            linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=16, num_experts_per_tok=4,
            moe_intermediate_size=32, shared_expert_intermediate_size=32, rms_norm_eps=1e-6,
            max_position_embeddings=128, full_attention_interval=4, tie_word_embeddings=False,
            hidden_act="silu", output_gate_type="sigmoid", hc_count=4, hc_lowrank=32,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=8, indexer_compress_ratio=4,
            ngram_size=3, ngram_vocab_size_base=97, heads_per_ngram=2, split_ngram_parts=1,
            make_ngram_vocab_size_divisible_by=32, ple_layer_ids=[2], ple_embed_dim=128, ple_conv_kernel_size=4,
            seed=1234, bos_token_id=2, eos_token_id=2,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_interleaved": True, "mrope_section": [2, 1, 1]})
TINY_SEED = 20261117
# The second prompt crosses an end of text (id 2), where the n-gram hash starts over, and both
# long ones have more complete blocks than QSA's budget of two.
TINY_PROMPTS = [[3, 17, 200, 45, 9],
                [11, 12, 13, 14, 15, 16, 17, 18, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [250, 128, 64, 32, 16, 8, 4, 99, 98, 97, 96, 95, 94, 93]]
TINY_GENERATE = 16
# The same layer mix with 256-wide attention heads, which the GPU kernels need, and a prompt
# long enough that most queries choose blocks; for checking the GPU path against transformers.
GPU = dict(TINY, head_dim=256, indexer_head_dim=64, rope_parameters=dict(TINY["rope_parameters"], mrope_section=[11, 11, 10]))
GPU_PROMPTS = TINY_PROMPTS + [[(7 * i + 3) % 250 + 3 for i in range(40)]]
SETS = (("qwen38_tiny", TINY, TINY_PROMPTS), ("qwen38_gpu", GPU, GPU_PROMPTS))
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden")


def write_tiny(out_dir, stem="qwen38_tiny", config=None, prompts=None):
    """A tiny Qwen3.8-Flash-Next (three Gated DeltaNet layers, then one QSA layer, the
    n-gram embedding at layer 1) with random weights, in llama.cpp's layout and
    weight types: Q8_0 matrices, BF16 indexer projections, an F16 n-gram
    convolution and a Q4_0 n-gram table. The reference runs on the dequantized values."""
    cfg = Qwen4ExpTextConfig(**(config or TINY))
    cfg.dtype = torch.float32
    prompts = prompts or TINY_PROMPTS
    assert cfg.layer_types == ["linear_attention"] * 3 + ["indexed_attention"]
    g = torch.Generator().manual_seed(TINY_SEED)
    Q8, Q4, BF16 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.BF16
    d, f, e, v = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.vocab_size
    hd, kh, vh = cfg.head_dim, cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    hc, low = cfg.hc_count, cfg.hc_lowrank
    conv_dim = 2 * kh * dk + vh * dv
    ple = cfg.ple_embed_dim
    probe = Qwen4ExpForCausalLM(cfg).model.layers[1].ple.ple_embedding
    consts = [probe.layer_multipliers.tolist(), probe.ngram_heads_offsets.tolist(), probe.ngram_heads_vocab_sizes.tolist()]
    table_rows = probe.ngram_embedding.weight.shape[0]
    del probe

    w = gguf.GGUFWriter(path=None, arch=ARCH)
    u32 = lambda key, x: w.add_uint32(f"{ARCH}.{key}", x)  # noqa: E731
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
    u32("expert_shared_feed_forward_length", cfg.shared_expert_intermediate_size)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(cfg.rope_parameters["rope_theta"])
    u32("rope.dimension_count", int(hd * 0.25))
    w.add_array(f"{ARCH}.rope.dimension_sections", cfg.rope_parameters["mrope_section"] + [0])
    u32("ssm.conv_kernel", cfg.linear_conv_kernel_dim)
    u32("ssm.state_size", dk)
    u32("ssm.group_count", kh)
    u32("ssm.time_step_rank", vh)
    u32("ssm.inner_size", vh * dv)
    u32("full_attention_interval", TINY["full_attention_interval"])
    w.add_array(f"{ARCH}.attention.recurrent_layers", [k == "linear_attention" for k in cfg.layer_types])
    u32("hyper_connection.count", hc)
    u32("hyper_connection.low_rank", low)
    u32("attention.indexer.head_count", cfg.indexer_n_heads)
    u32("attention.indexer.key_length", cfg.indexer_head_dim)
    u32("attention.indexer.top_k", cfg.indexer_budget)
    w.add_array(f"{ARCH}.attention.compress_ratios",
                [cfg.indexer_compress_ratio if k != "linear_attention" else 0 for k in cfg.layer_types])
    w.add_array(f"{ARCH}.ple.layers", [i - 1 for i in cfg.ple_layer_ids])
    u32("ple.ngram_size", cfg.ngram_size)
    u32("ple.heads_per_ngram", cfg.heads_per_ngram)
    u32("ple.conv_kernel", cfg.ple_conv_kernel_size)
    u32("ple.eos_token_id", cfg.eos_token_id)
    u32("embedding_length_per_layer_input", ple // ((cfg.ngram_size - 1) * cfg.heads_per_ngram))
    for key, x in zip(("layer_multipliers", "head_offsets", "head_vocab_sizes"), consts):
        w.add_key_value(f"{ARCH}.ple.{key}", x, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT64)

    tensors = {}
    stacked = {}

    def quant(name, x, kind):
        q = gguf.quants.quantize(x.numpy(), kind)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=kind)
        return gguf.quants.dequantize(q, kind).reshape(x.shape).astype(np.float32)

    def linear(name, shape, scale=1.0, kind=Q8):
        tensors[name] = quant(name, torch.randn(*shape, generator=g) * scale / shape[-1] ** 0.5, kind)

    def f32(name, x):
        w.add_tensor(name, x.numpy().astype(np.float32))
        tensors[name] = x.numpy().astype(np.float32)

    def gamma(name, n):
        f32(name, 1.0 + torch.randn(n, generator=g) * 0.1)

    def hyper(p, combine=True):
        gamma(p + "_norm.weight", hc * d)
        if combine:
            linear(p + "_down_inject.weight", (low + hc, hc * d))
        else:
            linear(p + "_down.weight", (low, hc * d))
        linear(p + "_up.weight", (hc * d, low))

    tensors["token_embd.weight"] = quant("token_embd.weight", torch.randn(v, d, generator=g), Q8)
    for l, kind in enumerate(cfg.layer_types):
        p = f"blk.{l}."
        hyper(p + "hc_attn")
        hyper(p + "hc_ffn")
        if kind == "linear_attention":
            linear(p + "attn_qkv_gate.weight", (conv_dim + vh * dv, d))
            linear(p + "ssm_beta_alpha.weight", (2 * vh, d))
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
            ih = cfg.indexer_head_dim
            linear(p + "indexer.q_proj.weight", (cfg.indexer_n_heads * ih, d), kind=BF16)
            linear(p + "indexer.k_proj.weight", (ih, d), kind=BF16)
            gamma(p + "indexer.q_norm.weight", ih)
            gamma(p + "indexer.k_norm.weight", ih)
        if l + 1 in cfg.ple_layer_ids:
            linear(p + "ple_key.weight", (hc * d, ple))
            linear(p + "ple_value.weight", (d, ple))
            for n in ("key", "query", "conv"):
                gamma(p + f"ple_norm_{n}.weight", hc * d)
            conv = (torch.randn(hc * d, cfg.ple_conv_kernel_size, generator=g) * 0.5).numpy().astype(np.float16)
            w.add_tensor(p + "ple_conv1d.weight", conv)
            tensors[p + "ple_conv1d.weight"] = conv.astype(np.float32)
        f32(p + "ffn_gate_inp.weight", torch.randn(e, d, generator=g) * (4.0 / d ** 0.5))
        f32(p + "ffn_gate_inp_shexp.weight", torch.randn(d, generator=g) / d ** 0.5)
        linear(p + "ffn_gate_shexp.weight", (cfg.shared_expert_intermediate_size, d))
        linear(p + "ffn_up_shexp.weight", (cfg.shared_expert_intermediate_size, d))
        linear(p + "ffn_down_shexp.weight", (d, cfg.shared_expert_intermediate_size))
        for gg, shape in (("gate", (f, d)), ("up", (f, d)), ("down", (d, f))):
            stacked[(l, gg)] = torch.from_numpy(quant(p + f"ffn_{gg}_exps.weight",
                                                      torch.randn(e, *shape, generator=g) / shape[-1] ** 0.5, Q8))
    hyper("output_hc", combine=False)
    linear("output.weight", (v, d))
    table = quant("per_layer_token_embd.weight", torch.randn(table_rows, ple // 4, generator=g), Q4)

    experts = lambda l: tuple((lambda x, s=stacked[(l, p)]: s[x]) for p in ("gate", "up", "down"))  # noqa: E731
    model = load(cfg, tensors, experts, lambda ids: table[ids], consts)
    path = os.path.join(out_dir, stem + ".gguf")
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out, gap = record(model, cfg, prompts, TINY_GENERATE)
    if not any(key.endswith("qsa_blocks") for key in out):
        sys.exit("no tiny prompt is long enough for QSA to choose blocks")
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
            for n in names:
                print(f"wrote tests/golden/{n} ({os.path.getsize(os.path.join(GOLDEN_DIR, n))} bytes)")
            print(f"smallest gap between the k-th and (k+1)-th router probability: {gap:.2e}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny(tmp, stem, config, prompts)
            bad = [n for n in names if digest(os.path.join(tmp, n)) != digest(os.path.join(GOLDEN_DIR, n))]
        if bad:
            sys.exit(f"MISMATCH: {', '.join(bad)} differ from a fresh run")
        print(f"OK: {names[0]} and {names[1]} match a fresh run")
    print(f"QSA block choices with a tie at the cut: {TIES.ties} of {TIES.choices} (kept the lower block)")


def main():
    if "--tiny" in sys.argv:
        return tiny_main("--check" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, nargs="+", help="the GGUF file, or each part of a split one")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0, help="also record this many tokens of greedy generate()")
    ap.add_argument("--long", help="also run a prompt made of this file's first --long-tokens tokens")
    ap.add_argument("--long-tokens", type=int, default=2600)
    ap.add_argument("--tail", type=int, default=32, help="logits kept for the long prompt's last positions")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    model, cfg = build(args.gguf, args.config)
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
    print(f"smallest gap between the k-th and (k+1)-th router probability: {gap:.2e}")
    print(f"QSA block choices with a tie at the cut: {TIES.ties} of {TIES.choices} (kept the lower block)")
    with open(args.out, "wb") as f:
        f.write(serialize(out))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes, {len(out)} tensors)")


if __name__ == "__main__":
    main()
