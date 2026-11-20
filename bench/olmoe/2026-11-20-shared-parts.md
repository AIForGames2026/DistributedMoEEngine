# Model families built from shared parts (2026-11-20)

TASK-42. One step loop now runs every family, each layer through the parts it has, and one table in `src/families.zig` is the only code that checks a GGUF architecture's name. Qwen2-MoE, a family the engine did not have, came from that table, a reference script and a tiny model, with no new step code.

## What changed

- **The table** (`src/families.zig`): one entry per GGUF architecture. Each reads its keys into a `Config` that describes the model by its parts, fills in what the file does not record (Mixtral's rope on adjacent pairs, gpt-oss's YaRN without truncation, Qwen2-MoE's top-k left unnormalized), and names the code that loads its non-expert weights. `Config.fromGguf` and `Weights.init` look the entry up.
- **The description** (`Config`, replacing the `Arch` enum):
  - **Attention:** QK norm (none, over the whole projection, or per head), and biases on the projections, the output and the router, plus sinks (`gptoss.Parts`, now loaded per flag).
  - **Rope:** whether YaRN scales cos and sin (false for DeepSeek, which folds it into the attention scale).
  - **Norms and blocks:** the feed-forward norm's tensor name, and the Gated DeltaNet's merged projections (Qwen3.8).
  - **Router:** softmax or sigmoid with a choice bias, with renormalizing as before.
  - **Experts:** biases and gpt-oss's SwiGLU.
  - **Names:** the family's name for messages.
- **The layer** (`olmoe.Layer`): GLM's and DeepSeek's parts moved onto it.
  - **In place of attention:** KDA, MLA with DSA, and MLA with decoupled rope.
  - **mHC:** its mixes before attention and before the feed-forward block.
  - **Feed-forward:** a dense block, or ungated shared experts.
  - **Router:** a sigmoid router's choice bias.
- **One step** (`olmoe.step`): every layer goes through the same five stages:
  - **Before each block:** the residual's norm, the hyper-connection mix, or mHC's mix and norm.
  - **Mixer:** GQA attention (now `Attention.run`), Gated DeltaNet, KDA, or either MLA.
  - **After each block:** the residual add, the hyper-connection inject, or mHC's new streams.
  - **Router:** softmax, or the sigmoid router that DeepSeek and GLM now share (`deepseek.route`; GLM's is the one-group case).
  - **Feed-forward:** dense, or experts plus gated or ungated shared experts.

  `glm53.step` and `deepseek.step` are gone. Buffers are sized from the parts the model has, so GLM and DeepSeek allocate no GQA buffers.
- **Lookahead for every family:** guesses skip dense layers, rank a sigmoid router's experts by score plus bias, and use mHC's mix for GLM.
- **GPU attention with biases:** a layer goes to the GPU's attention kernel unless it has sinks or a window, so Qwen2-MoE's biased attention runs there (gpt-oss's sinks still keep it on the CPU).
- **Architecture checks:** 22 before, in `olmoe.zig`, `olmoe_run.zig` and `expert_cache.zig`; none now outside `families.zig`. The parts' files read their own GGUF keys, and the n-gram embedding's keys take the file's own prefix.

## Line counts

| File | Before | After |
|---|---|---|
| `olmoe.zig` | 4,594 | 4,686 |
| `glm53.zig` | 1,577 | 1,390 |
| `deepseek.zig` | 635 | 495 |
| `gptoss.zig` | 125 | 130 |
| `qwen38.zig` | 437 | 437 |
| `families.zig` | none | 277 |
| `olmoe_run.zig` | 2,837 | 2,828 |
| Total | 10,205 | 10,243 |

Three step loops became one. The total grew by 38 lines with Qwen2-MoE's entry and test included.

## Same results, bit for bit

`record.sh` recorded each family's tokens and logits with the code before the refactor (commit 9d05193, built on `gx10-02`), and compared the refactored build with them (`--save`, then `--compare`). Each run prefills a fixed token sequence in chunks, then decodes greedily:
- **Tiny models:** 40 tokens prefilled 16 at a time, then 16 generated.
- **Real models:** 256 tokens prefilled 64 at a time, then 24 generated.

Each model ran four ways:
- one process on the CPU;
- one process with `--gpu`;
- split over three processes on the GPU with f32 activations;
- one process with `--lookahead 2 --extra 2`, compared with the run without it.

Running the old build twice gave the same bits, so the runs are deterministic.

| Model | CPU | GPU | Split | Lookahead |
|---|---|---|---|---|
| Tiny OLMoE, Qwen3-MoE, Qwen3.5, Qwen3.8, GLM-5.3, Mixtral, DeepSeek V3, gpt-oss | 8 of 8 | 8 of 8 | 8 of 8 | 8 of 8 |
| OLMoE-1B-7B (Q8_0) | same | same | same | same |
| Qwen3-30B-A3B (Q4_K_M) | same | same | same | same |
| Qwen3.6-35B-A3B (Q8_0) | same | same | same | same |
| Mixtral-8x7B (Q4_K_M) | same | same | same | same |
| Moonlight-16B (DeepSeek V3, Q4_K_M) | same | same | same | same |
| gpt-oss-20b (MXFP4) | same | same | same | same |
| GLM-5.3-Flash (Q4_K, experts streamed with `--cache 5000`) | not run | same | not run | same |

"Same" means every generated token and every logit matched the earlier build bit for bit. Qwen3.8-Flash-Next's GGUF is no longer on `gx10-02`, so its tiny models stand in for it. GLM-5.3-Flash ran only as TASK-32 ran it, on the GPU with its experts streamed, because its 170 GB do not fit in memory.

**Lookahead now covers DeepSeek and GLM.** Before, their steps made no guesses (recall 0.0%). Now, at two layers ahead, with outputs bit-identical with the predictor on and off:

| Model | Recall at H=2, prompts | Generation |
|---|---|---|
| Moonlight-16B | 81.8% (6,084 expert choices) | 82.2% (9,360) |
| Tiny DeepSeek V3 | 68.2% | |
| Tiny GLM-5.3 | 94.9% | |

**Real GLM-5.3-Flash with lookahead:** it now sends hints to the expert cache. 5.5% of expert calls found an expert a hint had loaded in time, against none before. That did not pay off here, though: the hints added 2,771 loads (21,955 against 19,184), and decode fell from 2.98 to 2.47 tokens/s. TASK-32's runs did not use lookahead, so its numbers stand.

**Speed** was not a goal. Decode in the GPU runs above was within 5% of before for every model except Moonlight, which went from 28.4 to 31.8 tokens/s (and from 3.4 to 5.7 on the CPU). Other jobs (a download, a conversion, and a reference run) shared the machine during both sets of runs, so these numbers are rough.

## The new family: Qwen2-MoE

Qwen1.5-MoE-A2.7B-Chat (14.3B parameters, 60 experts, top 4 plus a shared expert, GGUF architecture `qwen2moe`). It was the user's second choice: their first, Nemotron 3.5 Bolt, needs Mamba-2 layers, LatentMoE experts and a 2-bit format, all new math. The model is from `Qwen/Qwen1.5-MoE-A2.7B-Chat`, every file checked against Hugging Face's SHA-256, and converted on `gx10-02` with llama.cpp's `convert_hf_to_gguf.py` to Q8_0 (15.2 GB).

- **Built from:** the table entry (one line in the table and a 13-line function), which sets no QK norm, biases on the query, key and value projections, a gated shared expert, and top-k probabilities used as they are. Every one of those parts already existed.
- **Reference:** `tools/ref/ref_qwen2moe.py` loads the GGUF into transformers 4.57.1's `Qwen2MoeForCausalLM` in float32, with experts dequantized when chosen. `--tiny` writes `tests/golden/qwen2moe_tiny.{gguf,tfref}`: 2 layers, grouped KV heads, biases, a shared expert of a size unlike the experts', and Q8_0 matrices. `--tiny --check` reproduces both files byte for byte.
- **Tiny model:** in `zig build test` (135 tests on Windows, 139 in WSL2 and on `gx10-02` with RDMA and the GPU, all passing). Through `tfswarm` it matches transformers on the CPU, the GPU, and split over three processes: logits within 2.4e-6, and 48 of 48 generated tokens the same.

**Real model** against transformers (3 golden prompts, then 32 greedy tokens each). The smallest gap between a chosen expert and the next is 9.65e-7, yet every choice agrees. Logs are in `2026-11-20-shared-parts/logs/qwen2moe/`.

| Run | Logits, worst | Expert choices same | Top-1 | Generated tokens same | Decode |
|---|---|---|---|---|---|
| One process, GPU | 1.51e-4 | 3,552 of 3,552 | 37 of 37 | 96 of 96 | 38 to 42 tokens/s |
| One process, CPU | 1.64e-4 | 3,552 of 3,552 | 37 of 37 | 96 of 96 | 6.1 to 8.5 tokens/s |
| Three processes, GPU, f32 activations | 1.51e-4 | 3,552 of 3,552 | 37 of 37 | 96 of 96 | 30.5 to 31.9 tokens/s |
| One process, GPU, `--lookahead 2 --extra 2` | 1.51e-4 | 3,552 of 3,552 | 37 of 37 | 96 of 96 | recall 79.3% on prompts, 76.5% generating |

The decode rates include each prompt's prefill.

- **The split with fp8 activations** (its default) misses the 1e-2 bar: logits up to 2.84 off, and 97.5% to 98.6% of expert choices the same. Qwen3-30B-A3B misses it the same way (logits up to 13.8 off), so this is how fp8 compares with transformers, not something new for Qwen2-MoE. TASK-39 found the same for gpt-oss.
- **The fp8 split is not repeatable:** run twice with the same build, its logits differ by up to about 1.0, though all 24 tokens agree. That holds for the build before this task too (`qwen3-shm3-fp8` runs). The f32 split repeats bit for bit, which is why the comparisons above use it.

- **Tokenizer:** the GGUF's tokenizer gives the same ids as Hugging Face's on all 5,038 texts (167,338 tokens) of `check_tokenizer.py`, and decodes them back.
- **Chat template:** the GGUF's own ChatML template renders as transformers does on all 140 prompts, with identical ids (`check_chat.py --gguf-template`).

- **Through the server** (`--gpu --serve`, prompts from the GGUF's chat template):
  - **Capital of France:** answered "Paris" and stopped at `<|im_end|>`, at 51 tokens/s.
  - **A system turn and a question:** a coherent answer at 51 tokens/s, reusing the 4 prompt tokens it shared with the first request (`chat1.json`, `chat2.json`).

## Not done

- **Qwen3.8-Flash-Next's real model:** its GGUF is no longer on `gx10-02`, so only its two tiny models were compared.
- **GLM-5.3-Flash on the CPU and split:** too large for those runs here, as in TASK-32.

- **Out of scope:** new GPU kernels. MLA (DeepSeek's and GLM's), sinks and windows still run on the CPU.
- **The families needing new math:** Llama 4, Jamba, DeepSeek V2's router and Nemotron 3.5 Bolt.
- **Qwen2-MoE's dense layers:** `decoder_sparse_step` and `mlp_only_layers` are not in the GGUF. Every released Qwen2-MoE has none, and the reference refuses a config that does.
