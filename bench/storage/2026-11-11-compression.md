# Compressed expert streaming: why TASK-13 is cut (2026-11-11)

TASK-13 planned to store experts losslessly compressed on NVMe and decompress them on the GPU, and to turn compression on by default only if it made experts ready at least 10% sooner than raw reads. Before building it, `compress_study.py` ([`2026-11-11-compression/`](2026-11-11-compression/)) measured how far real expert weights can compress. Run it with `python3 compress_study.py MODEL.gguf`, standard library only. The output is in `compress_study.log`.

## Setup

- **Weights:** allenai's OLMoE-1B-7B-0924-Instruct Q8_0 GGUF, the same file as [`gx10-03-2026-11-05.md`](gx10-03-2026-11-05.md).
- **Sample:** 53.5 MB of Q8_0 blocks from all 48 expert tensors, four places in each (experts 0, 16, 32, and 48).
- **NVFP4:** 786,432 values from the same tensors, dequantized and requantized to NVFP4 (E2M1 values, one E4M3 scale per 16 values, a tensor scale). No NVFP4 MoE checkpoint the engine runs was at hand, and the engine has no NVFP4 GPU kernel.
- **Bounds:** "best order-0 ratio" is the limit for any coder that compresses each byte (or 4-bit code) on its own. It is computed from measured entropy, so it is what an ideal ANS or Huffman coder, as nvCOMP offers, can reach. zlib and lzma show what real general-purpose coders get.

## Results

| | Information per value | Best order-0 ratio | lzma | Best-case load time saved |
|---|---|---|---|---|
| Q8_0 | 7.68 of 8 bits (scales: 6.87 and 2.36 of 8 per byte) | 1.066 | 1.056 (values and scales split) | 6.2% |
| NVFP4 | 3.90 of 4 bits (scales: 3.70 of 8) | 1.090 | 1.072 (codes and scales split) | 8.3% |

zlib at level 9 on raw Q8_0 blocks gets 1.036, and lzma on raw blocks 1.029.

"Best-case load time saved" assumes decompression costs nothing and overlaps the read fully, so only fewer bytes cross the drive. Both formats stay below the 10% bar even then. Real decompression on the SMs would also compete with the expert matmuls, and the hardware Decompression Engine is absent on GB10 ([`../platform/gx10-03-2026-10-27.md`](../platform/gx10-03-2026-10-27.md)).

## Why so little

Quantization already removed most of the redundancy. Q8_0's 8-bit values use 7.7 of their 8 bits, and NVFP4's codes use 3.9 of their 4. Only the block scales compress well, and they are 6% of a Q8_0 block and 11% of an NVFP4 block.

## Decision

TASK-13 is cut (`cut-1`) and listed as future work in TASK-19. Expert-streaming speed would also matter only when experts are read from the drive (`--cache N`, TASK-11 and TASK-14); with every expert in memory, the default, compression gains nothing. If streaming needs to get faster, the larger levers are different:
- keeping more experts in memory;
- striping reads across more than one drive;
- lower-bit quantization, which changes output and so falls outside exact mode.
