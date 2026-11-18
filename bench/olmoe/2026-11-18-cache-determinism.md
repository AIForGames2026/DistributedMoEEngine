# Cached GPU expert runs, made repeatable bit for bit (2026-11-18)

Found while recording TASK-41's chats: with experts served through the expert cache (`--cache`) on the GPU, two identical runs gave the same tokens but slightly different logits, and two chats parted at a near-tie. Without the cache, GPU runs were identical. TASK-11 had shown that results do not depend on the cache, on the CPU only.

## Cause

Two separate differences, both on the GPU path only:

- **Kernel choice moved with load timing.** The cache path runs the experts that are ready in one GPU job, then the rest as they load (`src/split.zig`, `runGrouped`). `tf_gpu_experts` picks its kernels from the largest number of rows any expert has in that job: matrix-vector kernels for up to 8 (`MV_TOK`), tensor-core tiles above that (and for Q4_K, a wide kernel past 8). So the same expert could run on a different kernel, and round differently, depending on which experts happened to be ready together.
- **The outputs were added in another order.** Without the cache, the GPU's combine sums each token's weighted expert outputs with fused multiply-adds and adds that sum to the residual. The cache path added each weighted output into the residual on the CPU, one at a time.

## Fix

First version (later replaced by per-expert kernel choice, below):

- `ExpertJob` carried `most`, the largest count of any expert in the whole layer, which `runGrouped` computed before splitting the layer into jobs; `tf_gpu_experts` took the larger of it and its own count, so every job of a layer picked the kernels a single job would.
- With a GPU, `addInOrder` sums each token's calls first with fused multiply-adds, then adds the sum to the residual, as the GPU's combine does. It serves both the cache path and rank 0 of a split. CPU-only runs keep the old order, so TASK-33's bit-for-bit CPU results stand.

## Checks on `gx10-02`

Qwen3-30B-A3B, 1,024 tokens of `repo.tokens` then 128 generated, `--gpu --max-step 2048`, cache of 3,000 slots, compared with `--save` and `--compare` (logs in `2026-11-18-cache-determinism/`):

| Run, against the first | Before (`before-*.log`) | After (`after-*.log`) |
|---|---|---|
| Cache, again | 128 tokens, logits up to 6.1e-5 apart | identical |
| Cache with `--lookahead 1` | 128 tokens, up to 7.9e-4 apart | identical |
| Cache against no cache | 128 tokens, up to 5.3e-4 apart | identical |

The "after" rows compare against an all-in-memory GPU run (`after-d0.log`). With the kernel fix alone, cached runs already matched each other, but differed from the all-in-memory run by 1.35e-4 (`after-d.log`) until the combine order was matched too.

**Speed** (same command, cache of 3,000 slots, two runs each, `t41-*.log` before and `t42-*.log` after): decode 18.87 and 18.78 tokens/s before, 18.98 and 18.78 after; first token 6.72 and 5.89 s before, 5.82 and 5.78 s after.

**Goldens with the fix:**
- `zig build test`: 116 of 116.
- Qwen3-30B-A3B through the cache passes, logits within 2.51e-4 (`qwen3-cache.log`).
- OLMoE-1B-7B split over shared memory with `--gpu --payload f32` passes, as before (`olmoe-shm32-t42.log`). With the default fp8 payload it fails with or without the fix.
- GLM-5.3-Flash: the long prompt's top-1 at position 2598 now differs. Transformers' own top two there are 0.005 apart, while the long prompt's logits differ from transformers by 3.7. The user chose to excuse long-prompt top-1 differences where transformers' top two are less than 0.1 apart (TASK-32's amended criteria); the next closest position is 0.059 apart and every other one over 0.15. With that, it passes forward and generation (`glm53-tie.log`). The golden check now prints each differing top-1 with transformers' gap there.

## Follow-up the same day: kernels chosen per expert

The layer-wide count above could not reach an expert rank: `serveGrouped` gathers calls from rank 0 until its queue is idle, so its jobs never see a layer's full count. The user chose the general fix instead, which replaces the layer-wide count:

- `tf_gpu_experts` picks each expert's kernels from that expert's own row count: up to 8 rows (`SMALL_ROWS`, the old `MV_TOK` cut) the matrix-vector kernels (Q8_0) or the 8-row kernel (Q4_K), above that the tiled or wide kernels. A job holding both kinds launches both, the small experts' on the first part of the expert table and rows, the rest on the second. No kernel changed. How calls are grouped into jobs can no longer change any expert's rounding, on rank 0 or an expert rank.
- Callers list the small experts first (`gpu_small_rows` in `src/olmoe.zig`): `LocalExperts.run` lays out its rows in two passes, and `runBatch` partitions its held experts. A job that is not ordered falls back to the old rule for the whole job instead of failing.
- Grouping itself is unchanged: ready experts still go to the GPU together.

This changes all-in-memory rounding too: an expert with 8 or fewer rows in a prefill whose largest expert has more now runs on the matrix-vector kernels.

**Checks** (build `t43`, logs in `2026-11-18-cache-determinism/per-expert/`), the same Qwen3-30B-A3B run against an all-in-memory GPU run of the new build:

| Run | Result |
|---|---|
| All in memory, again | identical |
| Cache of 3,000 slots | identical |
| Cache with `--lookahead 1` | identical |
| Two processes over shared memory (`--payload f32`), twice | identical |

The two-process split matched with the previous build too (`shm42-*.log`), so this test never caught the expert rank mixing kernels; the change rules it out rather than fixing an observed difference.

**Speed**, decode at 1,024 tokens of context: all in memory, 36.24 tokens/s on average over five runs against 36.43 for the previous build (whose own runs spread from 35.98 to 36.58); with the cache, 18.74 and 18.75 against 18.51 and 19.11. First token unchanged (about 1.0 s in memory, 6.6 s through the cache).

**Goldens** (`per-expert/*.log`): OLMoE-1B-7B in one process and over shared memory (`--payload f32`), Qwen3-30B-A3B in memory, through the cache and with YaRN, and GLM-5.3-Flash (forward with its one near-tie, and generation) all pass. Qwen3.6-35B-A3B passes forward, but its generation check fails at step 3 (logits 0.335 apart, all 32 tokens the same), with this build and both earlier ones (`qwen36-t41.log`, `qwen36-t42.log`). That step is the exact router tie in transformers recorded in TASK-29; its golden file predates TASK-31's router gaps (`gen_gaps`), so the near-tie rule cannot excuse it. Regenerating that golden is left as its own task.
