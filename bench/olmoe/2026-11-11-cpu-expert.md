# CPU-only expert node on the desktop (TASK-22), Nov 11

OLMoE-1B-7B Q8_0 split between two machines:
- **Rank 0** (attention, router, LM head, and experts 0 to 31) ran on the GB10 GPU in `gx10-03` (`--gpu`).
- **Rank 1** (experts 32 to 63) ran on the desktop's CPU, an i7-12700K with 20 threads, in WSL2. Its `olmoe` was built without `-Dgpu`, so it links no CUDA library.

The ranks talked over plain TCP on the direct campus path from TASK-21 ([`../transport/2026-11-05-direct-path.md`](../transport/2026-11-05-direct-path.md)). The Spark dials the desktop's campus address, and `fwd.py` on Windows relays the connection into WSL2, since `olmoe` does not build for Windows. Settings: F32 payload (exact), the three TASK-10 prompts, then 64 generated tokens after each. Script, versions, and logs are in [`2026-11-11-cpu-expert/`](2026-11-11-cpu-expert/).

**All CPU numbers are desktop CPU results.** They are not GB10 results.

## Acceptance

| Check | Target | CPU expert, run 1 | CPU expert, run 2 |
|---|---|---|---|
| Logits vs transformers | within 1e-2 | 3.72e-5 | 3.72e-5 |
| Top-1 tokens | identical | 43/43 | 43/43 |
| Experts chosen | (TASK-10 also checks) | 5,504/5,504 | 5,504/5,504 |
| 3 x 64 generated tokens vs `generate()` | (TASK-24 check) | all match, logits within 1e-2 | all match, logits within 1e-2 |
| Per-expert latency on the CPU | recorded | 1,794 us average (max 8,124 us) | 1,834 us average (max 7,534 us) |
| Generation speed | recorded | 4.7 to 5.1 tokens/s | 4.6 to 5.0 tokens/s |

Rank 1 reports its per-expert latency itself. It is the time spent inside each expert's FFN, timed separately for each of 17,169 calls per run. Tokens/s include each prompt.

## Compared with an expert on the desktop GPU

This is the same link and the same split, with rank 1 on the RTX 5070 Ti instead (`-Dgpu=true --gpu`). It passed the same checks.

| Rank 1 | Per expert call | Expert time in the run | Generation | Prompts (5 / 21 / 17 tokens) |
|---|---|---|---|---|
| i7-12700K CPU | 1.79 to 1.83 ms | 30.8 to 31.5 s | 4.6 to 5.1 tokens/s | 1.44 to 1.50 / 2.67 / 2.09 to 2.11 s |
| RTX 5070 Ti | 0.23 ms | 3.9 s | 11.8 to 12.4 tokens/s | 1.69 / 0.67 / 0.61 s |

With the CPU expert, rank 1's FFN time is about 31 s of the roughly 40 s spent generating, so the CPU is now the limit, not the network. The CPU's expert FFN is 8x slower than the GPU's, and whole-run generation is about 2.5x slower.

## The first runs were 3.7x slower because of WSL2's localhost relay

The first three runs pointed `fwd.py` at `127.0.0.1`, which reaches WSL2 through its `localhost` relay. They passed every check but generated only 1.2 to 1.4 tokens/s, with the CPU expert (runs `cpu`, `cpu2`) and with the GPU expert (run `gpu`) alike. That is about 50 ms per layer, while rank 0 alone needs about 3 ms. So the time was in the path, not in either expert.

- **Payload size did not explain it.** A BF16 run sends one packet per expert call instead of two. It was only a little faster, and BF16 fails the 1e-2 logit target anyway (`diag-bf16.sh`).
- **Single-packet latency did not explain it.** `dispatch --latency` from `gx10-03` gave the same median through the relay as into native Windows: 794 us against 788 us one way. The relay path did have a long tail: p99 23.5 ms against 1.2 ms (`diag-link.txt`).
- **Skipping the relay fixed it.** Pointing `fwd.py` at the WSL2 VM's own address (`172.26.67.168` that boot) brought the same CPU run to 4.7 to 5.1 tokens/s (`diag-vmip.sh`, `diag-vmip2.sh`).

So rank 1 in WSL2 should be reached through its VM address, not through `localhost`. The tables above use those runs. The VM address changes when WSL2 restarts, so `run.sh` takes it as `FWD_TO`.

## How the CPU backend works

There is no separate CPU backend. A process with no GPU runs the same Q8_0 expert code on the CPU, so the backend is chosen per process: build without `-Dgpu`, or leave out `--gpu`. The placement manifest and the tracker do not need to know which one a process uses. The plan also called for dequantizing to FP32 at load time. That was not done: it would take 4x the memory (13.7 GB instead of 3.4 GB for rank 1's experts) and read 4x the bytes per call, with no gain in accuracy, since Q8_0 is already computed exactly against its scales. OLMoE has no MXFP4 or NVFP4 experts, so this run exercised only Q8_0. The loader's MXFP4 and NVFP4 decoding (TASK-08) is checked in the unit tests.

## Limits

- Each expert call runs on one thread, so the desktop's 20 threads are mostly idle. Running a layer's calls in parallel, or splitting each matmul across threads, is the obvious next step if CPU nodes matter.
- `olmoe` does not build for native Windows, so a Windows CPU node runs it in WSL2. Native Windows covers only the PoC's synthetic MoE (`dispatch.exe --moe`).
- The direct path has the limits TASK-21 recorded: the Spark must dial the desktop, and the path is not approved by campus IT.
