# OLMoE-1B-7B lookahead predictor (2026-11-10)

TASK-11 first results: an expert predictor that guesses two layers ahead, measured on two datasets in transformers and inside the engine on the desktop and over RDMA on `gx10-03`, and an expert cache that loads experts from the model file when the predictor hints at them.

## The predictor

After layer S's attention, rank 0 applies layer S + 2's own MoE norm and router to S's hidden state and keeps the top 8 + 2 experts per token as layer S + 2's guess. Layers 0 and 1 cannot be guessed that way, so they are guessed from the token embeddings, before layer 0's attention. It needs no training. The real router still picks the experts that run, so the predictor can only make loads start early.

In the two-process engine, each guess becomes a prefetch hint to rank 1 for the experts it owns (`flags` bit 6, `docs/wire-format.md`). Without `--cache`, every expert is loaded at startup and the hints are only timed; with it, they load experts (see "Expert cache" below).

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

## In the engine (`gx10-03`, RDMA)

GB10, `olmoe` built there ReleaseFast with `-Drdma=true -Dgpu=true`, two processes over RDMA loopback, F32 payload, `--generate 64 --lookahead 2`, every expert loaded at startup. The co-tenant used 106 of 121 GB throughout.

| Phase | Recall at H=2 | Layer 0 recall | Guessing time / layer time | Predictor on vs off |
|---|---|---|---|---|
| Golden prompts | 82.6% | 21.2% | 0.44% | bit-identical |
| Generation (3 x 64 tokens) | 83.6% | 22.5% | **0.56%** | bit-identical (all 192 tokens match `generate()`) |

Rank 1 saw 3,116 layer starts, each after its hint, by 6.7 ms on average and 0.26 ms at least. Speed is unchanged from TASK-24's run: 21.0, 18.4, and 19.0 tokens/s.

## Expert cache

`--cache N` holds a process's experts in N slots of about 6.7 MB instead of loading all of them at startup. A worker thread reads an expert through the io_uring O_DIRECT reader when a hint names it, or on demand, ahead of queued hints, when a call needs one that is not loaded. The least recently used slot is replaced, and layer 0 stays loaded (`--pin 1`) because its guesses are weak. Each expert's CRC32C is checked the first time it is read. On the desktop's discrete GPU the cached experts run on the CPU, which is why the speeds below are low; on GB10 the GPU reads the slots in place.

One process, 256 slots (1.72 GB, a quarter of OLMoE's 1,024 experts), 2-minute run of the three prompts' 64-token generations:

| Hints | Already loaded | Loaded by a hint in time | Hinted but still loading | Missed (loaded on demand) | Time waiting for loads | Loads (for hints) | Throughput |
|---|---|---|---|---|---|---|---|
| Off (`--lookahead 0`) | 52.9% | | | 47.1% | 52.5 s | 27,973 (0) | 2.0 tokens/s |
| On (`--lookahead 2`) | 49.4% | 41.3% | 0.0% | **9.3%** | **10.4 s** | 43,591 (38,062) | 2.7 tokens/s |

Of 59,392 expert calls each, hints cut misses by a factor of 5 and time waiting for loads by 5. Hints cost 56% more reads, and 13,474 hinted experts (35% of hint loads) were replaced before use. Tokens were identical every round and resident memory stayed flat. Loads took 1.85 ms each (6.7 MB from the WSL2 disk image, about 3.6 GB/s).

Every check still passes with the cache: in one process (256 slots) and in two over shared memory (160 slots on each rank), the prompts match transformers, all generated tokens match `generate()`, and output is bit-identical with the predictor on or off.

### Expert cache on `gx10-03` (GPU experts, RDMA)

Two processes over RDMA, each with its own cache of its 512 experts, 2- to 3-minute runs of the three prompts' 64-token generations. Every run's tokens were identical in every round, and resident memory stayed flat. Figures are rank 0's; rank 1's were within 2 points.

| Slots per rank | Hints | Already loaded | Hint in time | Hint still loading | Missed | Throughput |
|---|---|---|---|---|---|---|
| 128 (25%) | off | 54.3% | | | 45.7% | 8.0 tokens/s |
| 128 | top 8 + 2 | 50.8% | 22.0% | 10.4% | 16.8% | 7.9 tokens/s |
| 128 | top 8 | 51.3% | 24.8% | 9.2% | 14.8% | **9.0 tokens/s** |
| 256 (50%) | off | 79.1% | | | 20.9% | 12.3 tokens/s |
| 256 | top 8 + 2 | 71.5% | 15.9% | 5.0% | 7.6% | 12.6 tokens/s |
| 256 | top 8 | 74.3% | 15.3% | 3.5% | 6.9% | **13.1 tokens/s** |
| 520 (all) | off | 100.0% | | | 0.0% | 20.4 tokens/s (13 rounds) |

Hints cut misses by about two thirds at both sizes, but throughput rises only 7 to 12%, and only with 8 guesses rather than 10. The drive is the limit: the worker reads one expert at a time at about 3.2 GB/s (2.0 to 2.2 ms per expert), and both ranks share one NVMe drive. With 10 guesses at 128 slots, the two ranks read about 1.15 TB in 3.2 minutes (6 GB/s, close to the drive's 7 GB/s), and 30% of hinted loads were replaced before use, so the extra reads cancel what the hints save. A cache that holds every expert runs as fast as loading them all at startup (20.4 tokens/s), so the cache itself costs nothing once loaded.

Fixed during this run: in cache mode rank 1 loads no weights at startup and so never turned on GPU matmuls; its experts ran on the CPU (6.5 tokens/s with every expert loaded). One earlier run, made while rank 1 was still on the CPU, saw round 2's tokens differ from round 1's; it did not happen again in the 7 runs above (60 rounds) or in a repeat of that run, and the cause is not known.

## Acceptance (TASK-11)

| Criterion | Target | Desktop | `gx10-03` (RDMA) |
|---|---|---|---|
| Hints issued before the target layer starts | 95% of steps | 100% (3,116 of 3,116) | 100% (3,116 of 3,116) |
| Recall at H=2 with k + 2 guesses, two datasets | 80% | 82.7% WikiText-2, 82.2% HumanEval (transformers) | 83.6% in generation |
| Predictor compute overhead | under 2% of layer time | 1.62% (one process), 1.33% (two) | 0.56% |
| Output with predictor on vs off | bit-identical | bit-identical | bit-identical |

## Not yet measured or done

- Reading several experts at once, dropping hints that are already stale, and loading the most likely guesses first (TASK-14's miss handling). These are what would turn the lower miss rate into throughput.
- Experts in two processes on two different machines, where each has its own drive.
- Layer 0's guesses are weak (19 to 26%), so the cache keeps layer 0 loaded by default.
