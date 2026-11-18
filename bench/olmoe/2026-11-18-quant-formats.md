# Every GGUF quant format on the CPU and GPU (2026-11-18)

TASK-36. The engine read 8 weight formats, and the GPU ran only f32 and Q8_0 matmuls plus Q8_0 and Q4_K experts. Everything else ran on the CPU, and nothing said so. It now reads 24 formats, runs every one but f32 on the GPU, and names any weight left on the CPU.

## What was built

- **CPU decoding** (`src/quant.zig`): ports of ggml's reference `dequantize_row_*` for Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS. Each keeps ggml's order of float operations. The IQ lookup tables (`src/iq_tables.zig`, `csrc/cuda/iq_tables.cuh`) are generated from ggml's `ggml-common.h` by `tools/ref/gen_iq_tables.py`. The ggml code is from llama.cpp commit `ffc87c3a`, MIT, listed in `THIRD_PARTY.md`.
- **GPU kernels** (`csrc/cuda/blocks.cuh`): one small decode function per format turns 8 weights into exact f16 codes, plus a scale and offset shared by their group. One matrix-vector kernel (a warp per row, for up to 8 inputs) and one tensor-core kernel (the same tiles as the tuned Q8_0 one) use it for every format, instantiated once per format. Q8_0, f32 and Q4_K keep their tuned kernels.
- **Experts:** gate and up can be in one format and down in another, as Q4_K_M's Q4_K and Q6_K are. A mix of formats, or Q8_0 with the SwiGLU clamp, takes the generic kernels. Each expert still picks its kernels by its own row count, as fixed earlier today (`2026-11-18-cache-determinism.md`).
- **No silent fallback** (`src/gpu.zig`): the first time a weight runs on the CPU, the run logs its format, shape and the reason (no kernel, not mapped to the GPU, or mixed formats across experts). A count by reason and format is printed at the end.
- **Tools:** `olmoe gpu-formats` checks every format's GPU matmul and experts against the CPU. `tools/ref/check_ggml_quants.py` checks ggml's own decoding against gguf-py's.

## Checks

**Decoding, bit for bit.** `tests/golden/quant_formats.gguf` holds a random 4x1024 tensor in each of 21 block formats, with every value finite. The `.tfref` beside it has gguf-py 0.19's decoding. The engine's decoders match it bit for bit in `zig build test`. ggml itself (`libggml-base.so` built from the same commit on `gx10-02`) matches gguf-py bit for bit on all 21 (`2026-11-18-quant-formats/ggml-check.log`), so the engine matches ggml too.

**Unit tests:** 118 of 118 on `gx10-02`, 100 plus 1 skipped on Windows.

**GPU against the CPU** (`olmoe gpu-formats` on `gx10-02`, `2026-11-18-quant-formats/gpu-formats.log`), largest absolute gap, limit 1e-3:

| Check | Shapes | Largest gap |
|---|---|---|
| Matmuls, 23 formats | 512x2048 at 1, 3, 8, 9, 40 and 130 inputs | 1.14e-5 (MXFP4) |
| Experts, each format alone | 8 experts of 512x1024, 4 per token, at 2, 5 and 40 tokens | 1.45e-4 (MXFP4, largest output 392) |
| Experts, mixed formats | Q4_K/Q6_K, Q4_K/Q5_K, IQ4_XS/Q6_K, Q3_K/Q5_K, IQ2_XS/Q4_K, Q8_0/Q6_K | 1.14e-5 (Q4_K/Q5_K) |
| Experts, Q8_0 with the clamp | as above, limit 0.05 | 5.42e-8 |

The test weights have scales set so outputs stay at realistic sizes. With random scales, some outputs reached 1e5 and their gaps passed 1e-3, even though they were within about 1e-7 of the output.

**Models.** Qwen3-30B-A3B from bartowski's GGUFs, as Q4_K_M (18.6 GB) and IQ4_XS (16.5 GB). The references are transformers on each GGUF's dequantized weights, made by `tools/ref/ref_qwen3.py --gguf ... --generate 32` (22 and 33 minutes, run at the same time). The two files' formats:

| Tensors | Q4_K_M | IQ4_XS |
|---|---|---|
| Experts' gate and up | Q4_K | IQ4_XS |
| Experts' down | Q4_K or Q6_K (24 layers each) | IQ4_XS |
| attn_q, token_embd | Q4_K | IQ4_XS |
| attn_k | Q8_0 | Q8_0 |
| attn_v | Q4_K or Q6_K | Q5_K |
| attn_output | Q5_K | Q5_K |
| output | Q6_K | Q6_K |

Both pass the TASK-10 and TASK-24 checks every way they were run (`goldens.sh`, logs in `2026-11-18-quant-formats/gold/`). Every top-1 token matches, and all 32 generated tokens match `generate()`:

| Run | Q4_K_M worst logit gap | IQ4_XS worst logit gap |
|---|---|---|
| GPU | 1.85e-4 | 1.15e-4 |
| CPU only | 1.54e-4 | 1.02e-4 |
| Two processes over shared memory, `--payload f32` | 1.85e-4 | 1.15e-4 |
| Cache of 3,000 slots | 1.85e-4 | 1.15e-4 |

None of the GPU runs logged a weight on the CPU, so every expert and every other matmul ran on the GPU.

**Existing goldens** with the same build, since Q4_K and bf16 matmuls outside the experts now go to the GPU: OLMoE-1B-7B (local, and shared memory with `--payload f32`), Qwen3-30B-A3B Q8_0 (local, cache, and YaRN generation), Qwen3.6-35B-A3B and GLM-5.3-Flash all pass as before. GLM-5.3-Flash has one top-1 difference at a transformers near-tie, which TASK-32's amended criteria excuse.

## Speed

Qwen3-30B-A3B on `gx10-02`, 1,024 tokens of `repo.tokens`, then 128 generated, `--gpu`. Each file was run twice; the second run is shown, and the two were within 2% (`speed.sh`, logs in `2026-11-18-quant-formats/speed/`):

| GGUF | Decode | Prefill | Weights | Peak memory | Experts' share of decode |
|---|---|---|---|---|---|
| Q8_0 | 35.7 tokens/s (28 ms per token) | 1,112 tokens/s | 32.5 GB | 32.2 GB | 1.7 s |
| Q4_K_M | 41.9 tokens/s (24 ms) | 766 tokens/s | 18.6 GB | 18.7 GB | 1.3 s |
| IQ4_XS | 48.2 tokens/s (21 ms) | 908 tokens/s | 16.5 GB | 16.6 GB | 1.0 s |

Decoding reads fewer bytes per token, so the smaller files are faster. Prefill is slower than Q8_0's because Q8_0 keeps its tuned tensor-core kernel, while the generic one also decodes weights: the experts take 0.7 s (Q4_K_M) and 0.5 s (IQ4_XS) of prefill against Q8_0's 0.4 s.

## Not done

- f32 experts stay on the CPU. No GGUF stores them that way.
- The generic kernels have not been tuned. Prefill for the new formats is 20 to 30% slower than for Q8_0.
