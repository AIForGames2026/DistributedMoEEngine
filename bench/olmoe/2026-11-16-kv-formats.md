# Compact KV cache formats (TASK-27, 2026-11-16)

`--kv-format f32|bf16|fp8` sets how the KV cache stores keys and values (`src/kv_format.zig`). f32 stays the default.

- **bf16:** each value rounded to nearest even, 2 bytes.
- **fp8:** E4M3 (3 mantissa bits, saturating at 448) with one power-of-two scale per KV head of each position, stored as one byte after the row (padded to 16 bytes). A value decodes as its E4M3 value times its scale, exactly.

How it works:

- Rows are encoded once, on the CPU, as they enter the cache (threaded for prefill chunks). The CPU path and every GPU attention kernel decode exactly the same numbers.
- Each GPU attention kernel is compiled once per format, so the f32 kernels are the same code as before. A first version chose the format at run time inside the kernels, and that slowed f32 prefill attention by 32%.
- The NVMe tier (TASK-12) and saved prefixes (TASK-28) carry the rows as bytes, so both work in every format. The format is part of a saved prefix's fingerprint.
- Golden checks with a compact cache report how far the logits move instead of failing at 1e-2.
- `--compare` now also says how many tokens match before the first difference.

Raw logs, the script that ran them, and versions are in [`2026-11-16-kv-formats/`](2026-11-16-kv-formats/).

Build:

- `gx10-02`: this change on commit `2ce1572` (`v0.2.0-alpha-30-g2ce1572-kv27`), ReleaseFast, `-Dgpu=true -Drdma=true`.
- The f32 reference build is commit `2ce1572` itself.
- Machine: kernel 7.0.0-1019-nvidia, driver 580.178.04, Zig 0.17.0, nothing else running.

## Size

| Format | OLMoE-1B-7B, bytes per token | Qwen3-30B-A3B, bytes per token | Qwen3 at 128k | Against f32 |
|---|---|---|---|---|
| f32 | 262,144 | 196,608 | 25.77 GB | 1x |
| bf16 | 131,072 | 98,304 | 12.88 GB | 2x |
| fp8 | 66,048 | 50,688 | 6.64 GB | 3.97x (OLMoE), 3.88x (Qwen3) |

Bytes per token are keys and values over every layer. At 128k, the process's peak resident memory went from 57.2 GB (f32) to 44.6 GB (bf16) and 38.5 GB (fp8). That includes the 32.5 GB of weights.

## Against transformers (TASK-24 prompts)

The golden checks run the reference prompts against transformers' logits, then generate greedily and compare with `generate()`:

- **OLMoE:** `olmoe_generate.tfref`, 3 prompts, 64 tokens each.
- **Qwen3:** `qwen3_yarn_generate.tfref`, YaRN 4, 3 prompts, 32 tokens each.
- **Both:** on `gx10-02`'s GPU.

| Model and format | Prompt logits, worst error | Prompt top-1 | Generated tokens equal to `generate()` | Logits error, through the first difference |
|---|---|---|---|---|
| OLMoE f32 | 4.24e-5 | 43 of 43 | 192 of 192 | 4.48e-5 |
| OLMoE bf16 | 1.13 | 43 of 43 | 192 of 192 | 1.45 |
| OLMoE fp8 | 4.00 | 38 of 43 | 66 of 192 (37, 29, 0) | 1.58 |
| Qwen3 f32 | 2.51e-4 | 37 of 37 | 96 of 96 | 1.18e-4 |
| Qwen3 bf16 | 0.756 | 36 of 37 | 90 of 96 (26, 32, 32) | 2.49 |
| Qwen3 fp8 | 1.80 | 35 of 37 | 60 of 96 (26, 32, 2) | 2.14 |

The f32 rows are unchanged from before this change: the committed build gives the same OLMoE numbers (`olmoe-ref.log`).

Counting tokens until the first difference is a harsh measure: one near-tie flips the rest of the generation. The desktop CPU shows this (`desktop-cpu.log`, i7-12700K): there, OLMoE bf16 matched only 18 of 32 tokens on prompt 2. The GPU matched all 64, because the two paths add in different orders.

**What FP8 loses is in the keys.** A temporary patch, not kept, rounded only the keys or only the values of an f32 cache through each format. It ran on the desktop's RTX 5070 Ti, OLMoE prompt logits against transformers:

| Rounded | Worst logits error | Prompt top-1 |
|---|---|---|
| keys and values to fp8 (same as `--kv-format fp8`) | 4.00 | 38 of 43 |
| keys to fp8 | 3.70 | 39 of 43 |
| values to fp8 | 2.50 | 43 of 43 |
| keys to bf16 | 0.805 | 43 of 43 |
| values to bf16 | 1.13 | 43 of 43 |

So keys in bf16 with values in fp8 would likely keep bf16's top-1 agreement at 2.67x smaller. That format is not built.

## Speed, Qwen3-30B-A3B with YaRN 4, all in memory

Repository text from `repo.tokens`, prefilled in 2,048-token chunks, then greedy decode. bf16 and fp8 are compared with the f32 run's tokens and logits (`--compare`). At 32k, each format ran twice, alternating; the two runs agreed within 0.6%.

| Context | Format | First token | Decode | Against f32 | Tokens equal to f32 |
|---|---|---|---|---|---|
| 32,640 | f32 | 50.4 s | 15.50 tokens/s | | |
| 32,640 | bf16 | 48.6 s | 22.0 tokens/s | +42% | 128 of 128 |
| 32,640 | fp8 | 50.4 s | 26.6 tokens/s | +72% | first 7 of 128 |
| 131,040 | f32 | 519.8 s | 5.56 tokens/s | | |
| 131,040 | bf16 | 497.1 s | 9.63 tokens/s | +73% | none (the first token differs) |
| 131,040 | fp8 | 520.0 s | 14.43 tokens/s | +160% | none (the first token differs) |

Decode reads every cached position for each new token, so a smaller cache decodes faster, and more so the longer the context. At 128k, both compact formats picked a different first token than f32. That token's logits differ by 0.73 (bf16) and 6.3 (fp8).

Attention kernels alone (`attbench`, 32 query heads and 4 KV heads, as in Qwen3):
- **Prefill, 2,048 new tokens after 16,384 cached, tensor-core kernel:** f32 34.9 ms, bf16 32.0 ms, fp8 34.6 ms. The committed build's f32 kernel takes 34.8 ms.
- **Decode, one token after 32,768 cached:** f32 0.83 ms. The committed build takes 0.85 ms.

On the desktop CPU (OLMoE, short prompts), all three formats decode at 5.4 tokens/s.

## Unchanged f32, saved prefixes, and the NVMe tier

- **f32 against the committed build** (`regress-*.log`):
  - At 32,640 tokens, all 128 generated tokens and their logits match bit for bit.
  - First token 50.3 s against 50.1 s; decode 15.55 against 15.51 tokens/s.
- **Saved prefixes** (`prefix-*.log`, 16k prompts):
  - In bf16 and in fp8, a full restore and a partial one (10,240 positions from a 12,000-position file) give exactly a cold run's tokens and logits.
  - Files shrink with the format: 1.61 GB in bf16 and 0.83 GB in fp8 at 16,384 positions, against 3.2 GB in f32.
  - A full restore takes 0.31 s (bf16) and 0.16 s (fp8), so the first token comes in 0.35 s and 0.20 s instead of 18 s.
- **NVMe tier** (`tier-*.log`, 32,640 tokens, window 8,192):
  - The run with the middle on NVMe gives exactly the all-in-memory run's 32 tokens and logits, in bf16 and in fp8.
  - Decode with the tier is 2.68 tokens/s (bf16) and 3.75 tokens/s (fp8), since it reads the middle back for every token.

## Tests

`zig build test`: 86 of 86 on WSL2 and 82 of 83 (1 skipped) on Windows. New tests:
- E4M3 and bf16 rounding, checked exhaustively and on random inputs.
- Rows of each format decode within their precision.
- On the tiny OLMoE and Qwen3 models, bf16 and fp8 logits stay close to f32 but not equal, and one token at a time through a compact cache gives the full pass's logits.
