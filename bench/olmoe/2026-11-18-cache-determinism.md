# Cached GPU expert runs, made repeatable bit for bit (2026-11-18)

Found while recording TASK-41's chats: with experts served through the expert cache (`--cache`) on the GPU, two identical runs gave the same tokens but slightly different logits, and two chats parted at a near-tie. Without the cache, GPU runs were identical. TASK-11 had shown that results do not depend on the cache, on the CPU only.

## Cause

Two separate differences, both on the GPU path only:

- **Kernel choice moved with load timing.** The cache path runs the experts that are ready in one GPU job, then the rest as they load (`src/split.zig`, `runGrouped`). `tf_gpu_experts` picks its kernels from the largest number of rows any expert has in that job: matrix-vector kernels for up to 8 (`MV_TOK`), tensor-core tiles above that (and for Q4_K, a wide kernel past 8). So the same expert could run on a different kernel, and round differently, depending on which experts happened to be ready together.
- **The outputs were added in another order.** Without the cache, the GPU's combine sums each token's weighted expert outputs with fused multiply-adds and adds that sum to the residual. The cache path added each weighted output into the residual on the CPU, one at a time.

## Fix

- `ExpertJob` carries `most`, the largest count of any expert in the whole layer, which `runGrouped` computes before splitting the layer into jobs. `tf_gpu_experts` takes the larger of it and its own count, so every job of a layer picks the kernels a single job would (`src/olmoe.zig`, `src/split.zig`, `src/gpu.zig`, `csrc/tf_gpu.c`, `include/tf_gpu.h`). Raising it is safe: the kernels read each expert's rows from `first[]` and stop past them.
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

## Not covered

An expert rank (`serveGrouped`) gathers calls from rank 0 until its queue is idle, so its GPU jobs can also mix experts by timing. Its jobs do not know the layer's full count, so a split across machines on the GPU may still vary in rounding between runs. Rank 0's own share is now fixed.
