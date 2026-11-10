# OLMoE-1B-7B lookahead predictor (2026-11-10)

TASK-11 first results: an expert predictor that guesses two layers ahead, measured on two datasets in transformers and inside the engine on the desktop. Not yet run on `gx10-03`.

## The predictor

After layer S's attention, rank 0 applies layer S + 2's own MoE norm and router to S's hidden state and keeps the top 8 + 2 experts per token as layer S + 2's guess. Layers 0 and 1 cannot be guessed that way, so they are guessed from the token embeddings, before layer 0's attention. It needs no training. The real router still picks the experts that run, so the predictor can only make loads start early.

In the two-process engine, each guess becomes a prefetch hint to rank 1 for the experts it owns (`flags` bit 6, `docs/wire-format.md`). Rank 1 records the hints but does not load anything with them yet: every expert is still loaded at startup.

## Recall on two datasets

`tools/ref/lookahead_study.py` runs transformers' OlmoeForCausalLM on the same Q8_0 weights (float32, CPU) in 256-token chunks and scores every token and layer. Recall is the share of the 8 chosen experts found among the 10 guesses.

| Dataset | Tokens | Engine scheme at H=2 (layers 1 to 15) | Post-attention guesses only, H=2 (layers 2 to 15) | Same at H=1 (layers 1 to 15) | Layer 0 from embeddings | Layer 1 from embeddings |
|---|---|---|---|---|---|---|
| WikiText-2 raw, test split (first 16,384 tokens) | 16,384 | **82.7%** | 85.8% | 91.2% | 25.8% | 39.4% |
| HumanEval prompts and solutions (first 16,384 tokens) | 16,384 | **82.2%** | 85.8% | 91.1% | 19.0% | 30.7% |

**PASS**: above 80% at H=2 on both. Guessing from each layer's output instead of its post-attention state gives 88.4% and 87.6% at H=2, but leaves the hint one MoE block less time. As a baseline, the previous token's own experts at the same layer find 43.2% and 55.4%.

Files: WikiText-2 from `huggingface.co/datasets/Salesforce/wikitext` (`wikitext-2-raw-v1/test-00000-of-00001.parquet`, SHA-256 `5f1bea06...6d91`), HumanEval from `github.com/openai/human-eval` (`data/HumanEval.jsonl.gz`, SHA-256 `b796127e...79ef`). They are kept outside the repo. The run took 31 minutes on the desktop CPU.

## In the engine (desktop)

WSL2 Ubuntu 24.04, RTX 5070 Ti, ReleaseFast with `-Dgpu=true`, `olmoe local|shm MODEL --golden olmoe_generate.tfref --payload f32 --gpu --generate 64 --lookahead 2`. The CPU was otherwise idle.

| Mode | Phase | Recall at H=2 | Layer 0 recall | Guessing time / layer time | Predictor on vs off |
|---|---|---|---|---|---|
| One process | Golden prompts (43 tokens) | 82.6% | 21.2% | 0.84% | bit-identical (2,163,072 logits) |
| One process | Generation (3 x 64 tokens) | 83.6% | 22.5% | **1.62%** | bit-identical (9,658,368 logits, all 192 tokens) |
| Two processes, shared memory | Golden prompts | 82.6% | 21.2% | 0.70% | bit-identical |
| Two processes, shared memory | Generation | 83.6% | 22.5% | **1.33%** | bit-identical |

All 192 generated tokens still match transformers' `generate()`. Each run repeats every prompt and generation with the predictor off and compares the bits. Speed is unchanged: 31.7, 25.4, and 27.6 tokens/s in one process (2.0 to 2.5 s per 64 tokens).

**Hints before their layer.** In the two-process run, rank 0 sent 3,120 hints. Rank 1 saw 3,116 layer starts (layers where it ran at least one expert), and every one came after that layer's hint: by 6.1 ms on average, 0.38 ms at least. Because hints are sent in line, before the layer they are for, on the same ordered link as the expert calls, 100% arrive first (target 95%).

## Acceptance (TASK-11)

| Criterion | Target | Result |
|---|---|---|
| Hints issued before the target layer starts | 95% of steps | 100% (3,116 of 3,116 layer starts on rank 1) |
| Recall at H=2 with k + 2 guesses, two datasets | 80% | 82.7% WikiText-2, 82.2% HumanEval |
| Predictor compute overhead | under 2% of layer time | 1.62% (one process), 1.33% (two processes) |
| Output with predictor on vs off | bit-identical | bit-identical, prompts and generation |

## Not yet measured

- Any of this on `gx10-03` over RDMA.
- Whether hints arrive early enough to load experts from NVMe: nothing loads on a hint yet. An OLMoE expert is about 6.7 MB, roughly 1 ms at the drive's 7 GB/s, and the shortest lead time seen here is 0.38 ms.
- Layer 0's guesses are weak (19 to 26%), so a cache should keep layer 0's experts loaded (about 430 MB).
