# 128k context with the KV cache's middle on NVMe (TASK-12), Nov 11

Qwen3-30B-A3B (Qwen's Q8_0 GGUF, 32.5 GB) ran in one process on `gx10-02`'s GB10 with `--gpu --yarn 4`. YaRN scaling by 4 over 32,768 native positions is what Qwen documents for 128k. The prompt is the repository's own text (92 files, 227,983 tokens) tokenized with Qwen's tokenizer (`tools/ref/make_tokens.py`), cut to 131,040 tokens. That leaves room for 32 greedy tokens, ending exactly at 131,072 positions. Prefill ran in 2,048-token chunks.

Script, versions, and logs are in [`2026-11-11-long-context/`](2026-11-11-long-context/). Both Sparks were otherwise idle; `gx10-01` was not used.

## Acceptance

| Criterion | Result |
|---|---|
| (1) TASK-10 and TASK-24 checks against transformers, with YaRN | Logits within 2.65e-4 (target 1e-2), top-1 37/37, every routing choice identical (14,208), all 3 x 32 greedy tokens match transformers' `generate()` with logits within 1.14e-4 |
| (2) 128k with only the sinks and the window in memory, identical to all in RAM | All 32 generated tokens and their logits match the all-RAM run bit for bit |
| (3) Offloaded part read at 5 GB/s or more during decode | 6.31 GB/s (747.96 GB over 31 steps) |
| (4) Decode speed and time to first token recorded, both ways | Below |

## 128k, all in RAM against offloaded

| | All in RAM | Middle on NVMe |
|---|---|---|
| KV cache in memory | 25.77 GB | 3.07 GB |
| Process resident memory, max | 57.3 GB | 35.2 GB |
| Time to first token (131,040 tokens) | 1,918 s | 1,947 s (+1.5%) |
| Decode, positions 131,040 to 131,071 | 0.230 s per token | 3.83 s per token (16.7x) |
| Read from NVMe during decode | | 24.1 GB per token at 6.31 GB/s |
| Attention waiting for the read, per token | | 3.60 s |
| Whole run: read, written | | 1,459 GB read, 24.1 GB written (2.80 GB/s) |

The offloaded run's in-memory cache is:
- the first 128 positions;
- a ring of 8,192 + 2,048 positions plus four spare tiles (the window, room for one chunk, and partial tiles at both ends);
- two staging buffers that each hold one layer's middle while attention reads it (about 1 GB together).

Without the staging buffers the positions in memory take 2.0 GB.

Decode is bound by the drive, as estimated when the criteria were rewritten. Every new token reads every cached position in all 48 layers, and at 128k the middle is 24.1 GB in f32, so 6.3 GB/s gives a floor of about 3.8 s per token. Prefetching hides the compute behind the read, not the other way round: attention waited 3.6 of the 3.83 s. Prefill barely notices. Each chunk reads the middle too, but the reads overlap the chunk's much longer compute, so the time to first token grew by 1.5%.

Compact formats (TASK-27) would cut the bytes read per token by 2x (BF16) or 4x (FP8), still with offloaded and in-RAM runs matching each other. Saved prefixes (TASK-28) would skip the half-hour prefill on repeated runs.

## Correctness of the pieces

- **Qwen3-MoE in the engine.** Grouped KV heads (32 query, 4 KV), per-head QK norm, renormalized top-8 routing, and YaRN, checked against transformers 4.57.1's `Qwen3MoeForCausalLM`.
  - The reference (`tools/ref/ref_qwen3.py`) keeps the experts in Q8_0 and dequantizes on use, so it needs 28 to 34 GB rather than the 122 GB of an all-f32 model.
  - With plain rope and with YaRN, the logits' largest errors are the same numbers (1.70e-4, 2.65e-4, 6.87e-5). Each sits at the first position, where rope does nothing. The two goldens differ by up to 2.5 in logits, and generation only matches transformers' YaRN `generate()` with YaRN on.
  - A tiny random Qwen3-MoE with YaRN, `head_dim` unlike hidden / heads, and 2 KV heads now runs in the unit tests (`ref_qwen3.py --tiny`).
- **GPU attention.** A CUDA kernel reads the KV cache in place (GB10 shares memory) through a table of 32-position tiles. It adds positions up in the same order wherever a tile lives: in memory, in the ring, or in a staging buffer read from NVMe. That is why offloaded and all-RAM runs are bit-identical.
  - OLMoE passes its goldens with attention on the GPU.
  - With `TF_GPU_ATTENTION=0`, attention runs on the CPU and gives the same tokens with slightly different logits, as a different order of sums should.
- **Offloader** (`src/kv_tier.zig`). Tiles that leave the window are written to a file with O_DIRECT. Each layer's middle is read back through the TASK-03 io_uring reader by a thread that reads layer L + 1 while layer L computes. Before the 128k run, it was checked bit for bit against all-RAM runs:
  - on OLMoE with 512- and 1,024-position windows on the desktop;
  - at 16k on `gx10-02`.

## Speedups made during the runs

The 128k runs used the second attention kernel below. The router and expert changes came after them, and were measured on the same 16k prefill on `gx10-02` (`speedup.sh`, output in `speedup.txt`):

| Build | Time to first token, 16k | Decode at 16k |
|---|---|---|
| First attention kernel (16k pilot) | 182.5 s | 0.114 s per token |
| Register-tiled attention kernel (`tf_attention_wide`), as in the 128k runs | 139.1 s | 0.088 s per token |
| Plus the threaded router and one GPU call per layer for the experts | 95.7 s | 0.079 s per token |

- **Attention.** The tiled kernel gives each thread 4 x 4 scores and 4 x 8 outputs on 64 x 64 tiles, with 128-bit shared loads laid out to avoid bank conflicts. `attbench` measured it at 4.6x faster than the first kernel with 16k cached positions and 5.9x with 64k. That was while the first 128k attempt was still using the GPU, which was then stopped and restarted on the new kernel. Its outputs differ from the first kernel's by at most 5e-8.
- **Router.** The f32 router matmul now walks the inputs in cache-sized blocks and splits rows across threads (`TF_THREADS`). That made it 4x faster on the desktop (1.2 to 0.3 s for a 4k OLMoE prefill). Each row is computed exactly as on one thread.
- **Experts.** Prefill groups each layer's calls by expert. One GPU call (`tf_gpu_experts`) copies every input in, queues gate, up, SiLU, and down for every expert, and copies the outputs out, instead of three matmuls per expert that each copy and wait. That made the experts 2.1 to 2.6x faster on the desktop (11 to 14 s down to 5.3 s). Moving SiLU to the GPU can change the last bit of results against earlier builds; runs of one build still match bit for bit, offloaded and all-RAM alike.

The profile after these changes (16k, GB10) still puts the experts first: 51 s of 96. Next come attention (13 s), the attention projections (11 s), and the output projection (10 s). The Q8_0 matmul kernel takes 8 tokens per pass, so a 2,048-token chunk reads each weight 256 times. A tiled matmul for many tokens is the next prefill speedup. Prefill at 128k was not rerun with these changes.

## Limits

- One process: the split two-process modes cap a forward pass at 64 tokens, so long prefill is local only.
- GPU only: the tiered cache needs GPU attention with 128-wide heads. Without a GPU, the plain cache and CPU attention still work.
- f32 cache: the formats of TASK-27 are not implemented.
- The repository text is a convenient long prompt, not a benchmark of quality at long context.
