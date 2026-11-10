"""TASK-11 study: how well do untrained lookahead predictors guess OLMoE's
experts H layers ahead? Runs transformers' OlmoeForCausalLM on the Q8_0 GGUF
(as ref_olmoe.py does) and, for every token and layer pair (S, T = S + H),
scores the router of layer T applied to layer S's hidden state. Recall is the
share of the k true experts found among the top k + extra candidates.

  ref_olmoe.py-style args: --gguf --config --tokenizer, plus text inputs:
  --text FILE (repeatable; plain text, chunked) or the golden prompts.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_olmoe  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def chunks(tok, path, size, limit):
    text = open(path, encoding="utf-8").read()
    ids = tok.encode(text).ids
    out = [ids[i:i + size] for i in range(0, min(len(ids), limit), size)]
    return [c for c in out if len(c) > 8]


def study(model, cfg, seqs, extra, horizons):
    layers = cfg.num_hidden_layers
    k = cfg.num_experts_per_tok
    post_attn = [None] * layers
    rec = {"on": False}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        def grab(mod, args, l=l):
            if rec["on"]:
                post_attn[l] = args[0].detach().reshape(-1, cfg.hidden_size)
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(grab))
    # (source kind, H) -> [hits, total]
    score = {}
    tokens = 0
    with torch.no_grad():
        for ids in seqs:
            rec["on"] = True
            res = model(torch.tensor([ids]), output_hidden_states=True, output_router_logits=True, use_cache=False)
            tokens += len(ids)
            # Scoring below calls the norms again, which would fire the hooks.
            rec["on"] = False
            true = [torch.topk(res.router_logits[l].float(), k, dim=-1).indices for l in range(layers)]
            # Layer inputs: hidden_states[l] is what layer l starts from.
            inputs = [res.hidden_states[l][0] for l in range(layers)]
            for t in range(layers):
                # The previous token's router at the same layer, known long before.
                cand = torch.topk(res.router_logits[t].float()[:-1], k + extra, dim=-1).indices
                hit = (cand.unsqueeze(-1) == true[t][1:].unsqueeze(1)).any(1).sum().item()
                s = score.setdefault(("prev_token", 0), [0, 0])
                s[0] += hit
                s[1] += true[t][1:].numel()
            for h in horizons:
                # The engine guesses layer h - 1 from the token embeddings, before any attention.
                tl = model.model.layers[h - 1]
                cand = torch.topk(tl.mlp.gate(tl.post_attention_layernorm(inputs[0])), k + extra, dim=-1).indices
                hit = (cand.unsqueeze(-1) == true[h - 1].unsqueeze(1)).any(1).sum().item()
                s = score.setdefault(("embed", h), [0, 0])
                s[0] += hit
                s[1] += true[h - 1].numel()
                for t in range(h, layers):
                    tl = model.model.layers[t]
                    for kind, src in (("post_attn", post_attn[t - h]), ("layer_in", inputs[t - h + 1] if h > 1 else None)):
                        if src is None:
                            continue
                        logits = tl.mlp.gate(tl.post_attention_layernorm(src))
                        cand = torch.topk(logits, k + extra, dim=-1).indices
                        hit = (cand.unsqueeze(-1) == true[t].unsqueeze(1)).any(1).sum().item()
                        s = score.setdefault((kind, h), [0, 0])
                        s[0] += hit
                        s[1] += true[t].numel()
    for hk in hooks:
        hk.remove()
    return score, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--limit", type=int, default=4096, help="tokens per text file")
    ap.add_argument("--extra", type=int, default=2)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg = ref_olmoe.build(args.gguf, args.config)
    sets = {}
    if args.text:
        for p in args.text:
            sets[os.path.basename(p)] = chunks(tok, p, args.chunk, args.limit)
    else:
        sets["golden"] = [tok.encode(p).ids for p in ref_olmoe.PROMPTS]
    report = {}
    for name, seqs in sets.items():
        score, n = study(model, cfg, seqs, args.extra, (1, 2))
        report[name] = {"tokens": n, "extra": args.extra,
                        **{f"{kind} H={h}": round(a / b, 4) for (kind, h), (a, b) in sorted(score.items())}}
        for h in (1, 2):
            # What the engine scores: embedding guesses for the first layers, post-attention for the rest.
            a = score[("embed", h)][0] + score[("post_attn", h)][0]
            b = score[("embed", h)][1] + score[("post_attn", h)][1]
            report[name][f"engine H={h}"] = round(a / b, 4)
        print(json.dumps({name: report[name]}), flush=True)


if __name__ == "__main__":
    main()
