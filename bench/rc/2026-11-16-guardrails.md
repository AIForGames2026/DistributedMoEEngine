# Capacity and memory guardrails (TASK-17, 2026-11-16)

TASK-17: an engine process must survive a load spike without unbounded work, without overflowing its rings, and without growing past a set amount of memory on a shared machine. Raw logs and the scripts are in [`2026-11-16-guardrails/`](2026-11-16-guardrails/).

## What was added

**Capacity (`--capacity C`, off by default).**
- **The limit:** each process takes at most C times its even share of a nominal step's expert calls per layer. The nominal step is `--chunk`, 64 tokens for the split. With half of OLMoE's experts and C = 1.25, that is 1.25 × 64 tokens × 8 choices / 2 = 320 calls per layer.
- **What is dropped:** in any layer where a rank would get more, its calls with the lowest gate weights are dropped. Each such token keeps its residual path and its other experts. The drops are counted and reported.
- **Receiver-granted:** rank 1 works out its own limit and grants it to rank 0 as a control immediate (`gen8` 1, the count in `slot`; `docs/wire-format.md`). Rank 0 waits for the grant before its first step.
- **Ring slots:** the slot credits from TASK-02 and TASK-07 still decide when each packet may be written. A full ring makes the sender wait, never overwrite. The grant only bounds how much work one step can bring.
- **Per expert (`--expert-capacity C`):** the same limit applied to each expert, the form the task names. OLMoE's routing makes it the wrong tool (see below), so it is optional and not used in the runs.
- **Output:** dropping calls changes the output, so both limits are off unless asked for. Every earlier result is unchanged.

**Memory budget (`--memory-budget MB`).**
- **The heap:** every allocation goes through a counting allocator (`src/budget.zig`), including the KV cache and the batched-expert scratch, which used to come straight from the page allocator.
- **Arming:** once the weights are loaded and the GPU is set up, the budget is armed. Its limit is the budget minus what is already resident, plus GPU memory, which on the integrated GB10 comes out of system memory without showing in the resident set (`tf_gpu_device_bytes`). From then on the heap refuses any allocation that would pass the limit, and the step that wanted it fails instead of pushing the machine into swap.
- **Checks:** resident plus GPU memory is checked after every engine step, and on rank 1 every 256 expert calls.
- **Heap balance:** every process reports its heap at exit: allocations, frees, peak, and bytes still allocated.

**Spike test (`--spike F`).** With `--context`, the middle third of the prompt arrives in steps F times as big. Each phase (before, during, after) reports its steps, speed, longest step, drops, rank 0's longest wait on rank 1 with nothing sent or received, and memory. `--max-step` raises the split's 64-token step limit to make room for the spike.

## Why the limit is per rank

Capacity factors are usually per expert. OLMoE's router concentrates hard: within a 64-token chunk of ordinary text, a few experts are chosen by nearly every token. Two recordings of a 6,144-token run on the desktop (64-token and 192-token steps; `capacity_curve.py` computes both policies from their frames) give these drop rates:

| Factor | Per expert, nominal | Per expert, 3x spike | Per rank, nominal | Per rank, 3x spike |
|---|---|---|---|---|
| 1.0 | 46.1% | 74.6% | 6.68% | 66.7% |
| 1.25 | 38.9% | 69.4% | **0.47%** | 58.3% |
| 1.5 | 33.2% | 64.7% | 0.01% | 50.0% |
| 2 | 24.7% | 56.5% | 0% | 33.4% |
| 4 | 8.5% | 34.6% | 0% | 0.1% |
| 7 | 0.46% | 19.1% | 0% | 0% |
| 8 | 0% | 16.0% | 0% | 0% |

- **Per expert:** keeping nominal drops under 1% needs factor 7, which lets an expert take 56 of a possible 64 calls. That limit bounds almost nothing.
- **Per rank:** the load evens out over each rank's 32 experts. Rank 1's share of a layer's calls ranged from 23% to 80%, with a median of 47.5%. At the standard factor of 1.25, nominal drops are under 0.5%, and a 3x spike is held to 1.25 times a nominal step's work.
- **Decode:** a single token never drops under either limit.

## Runs

Two DGX Sparks over RDMA on their cable:
- **Rank 0:** `rdma-router` on `gx10-02`, otherwise idle.
- **Rank 1:** `rdma-expert` on `gx10-03`, beside the co-tenant's vLLM job (96 GB of GPU memory, about 15 GB of RAM free).

Settings: OLMoE-1B-7B Q8_0, F32 payload, GPU on both ranks, 4,032 prompt tokens in 64-token steps, with steps of 192 tokens (3x) for the middle third, then 32 decoded tokens.

Build: commit `d3c271c` plus this change, ReleaseFast, `-Drdma=true -Dgpu=true`; kernel 7.0.0-1019-nvidia, driver 580.178.04, Zig 0.17.0.

### The 3x spike, with and without the guardrails

| | Without (7 runs) | With `--capacity 1.25 --memory-budget 7000` (4 runs) |
|---|---|---|
| Spike steps (192 tokens) | longest 1,671 to 1,718 ms; 114 to 117 tokens/s | longest 661 to 700 ms; 289 to 310 tokens/s |
| Nominal steps (64 tokens) | longest 508 to 560 ms after the first step | longest 516 to 551 ms |
| Calls dropped, before / during / after the spike | none | **0.179%** / 58.3% / **0.224%** (308 / 100,352 / 386 calls) |
| Calls sent to rank 1 | 250,772 | 203,244 |
| Rank 0's longest wait on rank 1, during and after the spike | at most 1.2 ms | at most 0.6 ms |
| Rank 0's longest wait on rank 1, whole run | 0.9 to 12.6 ms in 5 runs; **2,250 ms and 336 ms** in 2 (see below) | 1.0 to 2.3 ms |
| Memory, rank 0 / rank 1 | 5,169 MB / 3,545 MB | at most 5,170 MB / 3,545 MB of 7,000, 0 checks over, 0 allocations refused |
| Heap at exit, rank 0 / rank 1 | 289 / 36 allocations, all freed | 289 / 36 allocations, all freed (**0 bytes outstanding**) |

- **Without the guardrails:** the spike steps take three times as long as nominal ones, since every call runs.
- **With them:** each rank's work per layer is held to 320 calls, so a spike step costs about 1.3 times a nominal step, and the drops fall almost entirely in the spike.
- **Ring slots:** sends waited for rank 1 to grant ring slots about 5,300 times without the guardrails and 3,900 with them, and never for long.
- **Results:** the drops happened in the prompt, so the decoded tokens differ from an exact run. That is expected; every run finished cleanly.

**Open issue: a slow first step.** In two runs without the guardrails, rank 0 once waited 2,250 ms and 336 ms on rank 1 before the spike began:
- **Where:** in the later runs, which label the step, every phase's longest wait before the spike was in step 1, and the largest was 12.6 ms. So the long waits were most likely in the first step of each run.
- **Not the network:** both NICs' retransmit and error counters (`local_ack_timeout_err`, `out_of_sequence`, `packet_seq_err`, `rnr_nak_retry_err`) read 0, so no packet was resent.
- **Not rank 1's compute:** its slowest expert call took 4.3 ms.
- **Likely cause, unconfirmed:** a cold start on `gx10-03`, which has about 15 GB free beside its co-tenant (first use of fresh memory, or of the GPU, by a new process).
- **Under load:** no wait under load exceeded 4.3 ms in any of the 13 runs, with or without the guardrails. A warm-up step before serving would hide the first-step wait; it is not added here.

### Memory budget refusing

| Run | Budget | Held at startup | Result |
|---|---|---|---|
| Split, as above | 4,500 MB | rank 0: 4,027 MB, leaving 442 MB | Rank 0's KV cache (1.07 GB) was refused. Rank 0 stopped with `OutOfMemory` (exit 1) at 4,058 MB, never over budget, heap balanced; rank 1 exited 0. |
| Desktop, OLMoE in one process (RTX 5070 Ti) | 8,000 MB | 7,476 MB | KV cache refused; stopped at 7,486 MB, heap balanced |
| Desktop, the same | 7,000 MB | 7,476 MB | Refused at startup: the weights alone are over budget |

### Qwen3-30B-A3B, one process on `gx10-02`

The same spike with `--capacity 1.25 --memory-budget 40000`:
- **Limit:** one process takes 1.25 × 64 × 8 = 640 calls per layer, so a nominal step (512 calls) never drops.
- **Drops:** none in 2,688 nominal tokens; 58.3% during the spike, whose steps took at most 384 ms against 354 to 380 ms for nominal ones.
- **Memory:** 32,573 MB was held at startup, at most 33,466 MB of the 40,000 MB budget, and the heap was balanced at exit (320 allocations).

### No change without the new options

The committed build (`d3c271c`) and this one, both ranks forked on `gx10-02` over RDMA, `--context 512 --chunk 64 --generate 256`, alternating:
- **Exactness:** all six runs matched one saved reference bit for bit (`--compare`), tokens and logits.
- **Speed:** decode 35.39 tokens/s with this build against 35.51 with the committed one (-0.3%); first token 9.28 s against 9.22 s. Both are within run-to-run noise.
- **Tests:** unit tests pass, 81 of 81 on WSL and 77 of 78 on Windows (one Linux-only test skipped). That includes new tests for the counting heap, both capacity limits, a forward pass with calls dropped on both expert paths, and the grant.

## Acceptance (TASK-17)

| Criterion | Result |
|---|---|
| A synthetic 3x load spike completes, with dropped-token counts reported | **pass**: every run completed; drops reported per phase and in total |
| No memory leak (allocator balance at zero after the run) | **pass**: 0 bytes outstanding in every process of every run |
| No pipeline stall longer than 100 ms | **pass under load**: at most 4.3 ms during and after the spike in all 13 runs; the first step of 2 unguarded runs waited 336 and 2,250 ms (open issue above) |
| The process never exceeds its configured memory budget | **pass**: 0 of 82 checks over on rank 0 and 0 of 795 on rank 1; too-small budgets were refused rather than exceeded |
| Token drop rate under 1% at nominal load | **pass**: 0.18% before the spike and 0.22% after, per-rank capacity 1.25 |

Departure: the description's per-expert capacity factor is here (`--expert-capacity`), but the limit that meets the targets on OLMoE is per process, granted by the receiver.
