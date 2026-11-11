# Saved prompt prefixes, Nov 11

TASK-28: `olmoe --context N --prefix-cache DIR` saves a prompt's KV cache after prefill and reads it back when a later prompt starts with the same tokens. Same setup as [the speedup runs](2026-11-11-speedups-2.md): Qwen3-30B-A3B (Q8_0) on `gx10-02`'s GB10 with `--gpu --yarn 4`, the repository's own text as the prompt, 2,048-token chunks. The prefix files went to the same NVMe drive. Scripts and logs are in [`2026-11-11-prefix-cache/`](2026-11-11-prefix-cache/) (`run.sh`, `console.log`).

## Results

| Prompt | Time to first token | Matches a cold run |
|---|---|---|
| 16,384 tokens, cold | 17.1 s | (reference) |
| 16,384 tokens, 10,240 restored from a saved 12,000-token prefix | 8.4 s | bit for bit |
| 16,384 tokens, all restored | 0.6 s | bit for bit |
| 12,000 tokens, all restored | 0.4 s | bit for bit |
| 131,040 tokens, cold | 509.7 s | (reference) |
| 131,040 tokens, all restored | 4.2 s | bit for bit |

"Matches a cold run" means the 8 generated tokens (32 at 128k) and the logits each was picked from are identical to a run without a prefix cache.

| Transfer | Size | Time | Rate |
|---|---|---|---|
| Save, 16,384 positions | 3.22 GB | 0.54 s | 5.9 GB/s |
| Restore, 16,384 positions | 3.22 GB | 0.53 s | 6.0 GB/s |
| Save, 131,040 positions | 25.76 GB | 4.40 s | 5.9 GB/s |
| Restore, 131,040 positions | 25.76 GB | 4.22 s | 6.1 GB/s |

A save happens after the first token, so it doesn't count toward time to first token, and decode speed is unchanged (0.046 s per token at 16k, 0.181 s at 128k).

At 128k a restored prompt reaches its first token 121x sooner than a cold one. Nearly all of the 4.2 s is reading 25.8 GB at 6.1 GB/s, close to the drive's 7 GB/s.

A run with `TF_MM_KERNEL=2` found no usable prefix, as intended: its fingerprint differs, because that kernel rounds differently.

The same checks passed first on the desktop (OLMoE, RTX 5070 Ti, 1,024-token chunks): a repeated 3,000-token prompt took 0.2 s to first token instead of 2.5 s, and every run, full or partial, matched its cold run bit for bit.

## How it works

- **File:** one per saved prompt, named by a hash of the fingerprint and the tokens. It holds a header with the fingerprint, the tokens, the logits of the last position, and each layer's keys and values (f32, 196,608 bytes per position for Qwen3-30B-A3B). Every part starts on a 4 KiB boundary, so saves use O_DIRECT writes straight from the KV cache and restores use the TASK-03 io_uring reader straight into it.
- **Lookup:** compares the prompt with the tokens of every file in the directory whose fingerprint matches, and takes the one that covers the most.
- **Why whole chunks:** the GPU picks kernels by batch size, so a position's keys and values depend on the chunk it was prefilled in. A prompt identical to a saved one takes everything, logits included, and skips prefill. Otherwise a prompt takes the largest whole number of chunks it shares with the file, leaving at least one position to prefill, and prefills the rest with the same chunk boundaries as a cold run. That is what keeps output bit-identical.
- **Fingerprint:** the engine build (`git describe`, or `-Dengine-build`), the model file's name and size, `--yarn`, `--gpu`, `--chunk`, `TF_ATT_KERNEL`, and `TF_MM_KERNEL`. A file with any other fingerprint is ignored.

## Limits

- Only the in-memory KV cache. With `--kv-offload` the cold middle sits in the tiered cache's own file, and restoring into it would go through its ring buffer instead of reading straight into memory.
- Files are never deleted automatically, and a prompt that extends a saved one is saved again in full: 25.8 GB per 128k prompt. Compact KV formats (TASK-27) would make files 2 to 4 times smaller and restores that much faster.
- Only `olmoe --context` uses it so far; the golden and generation modes start from an empty cache.
