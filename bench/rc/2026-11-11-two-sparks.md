# Two-Spark RDMA run (TASK-15b, 2026-11-11)

TASK-15b acceptance run: one engine process per DGX Spark, rank 0 (attention, router, LM head, experts 0 to 31 of each layer) on `gx10-02` and rank 1 (experts 32 to 63) on `gx10-03`, over RDMA on the direct cable between them (`rocep1s0f0`, 192.168.197.12, to `rocep1s0f1`, 192.168.197.13; RoCE v2, GID index 3, MTU 4096). `gx10-01` was running another job, so this is two nodes, not three.

Raw logs, load samples, versions, and the scripts that ran them are in [`2026-11-11-two-sparks/`](2026-11-11-two-sparks/). Builds: ReleaseFast on each machine, `dispatch` with `-Drdma=true -Dcpu=native`, `olmoe` with `-Drdma=true -Dgpu=true`; commit `f07b91a` for the dispatch checks and run 1, plus the soak diagnostics added for run 2. Both machines: kernel 7.0.0-1019-nvidia, driver 580.178.04, Zig 0.17.0. `gx10-02` was otherwise idle; on `gx10-03` the co-tenant's vLLM job held about 96 GB of GPU memory and 106 to 116 GB of RAM throughout.

## TASK-07 suite across the two nodes

| Check | Result |
|---|---|
| Unit tests (`zig build test -Drdma=true`) on each machine | 45 of 45 on both |
| TASK-02 dispatch check (4,096 + 4,096 packets, `dispatch rdma-expert` / `rdma-router`) | 20 of 20 pass, both processes exit 0; 3.61 to 3.68 us per token pipelined (loopback on `gx10-03`, Nov 5: 2.76) |
| PoC MoE check (`--moe`) | 20 of 20 pass: routing identical, logits max error 7.15e-7, top-1 16/16, about 0.6 ms total |

## Against `perftest` (same cable, 7,552-byte messages)

| Measure | `perftest` | Target | Ours | Result |
|---|---|---|---|---|
| One-way latency, p50 | `ib_write_lat` typical 3.34 to 3.41 us (3 runs of 100,000) | at most 1.5x, about 5.1 us | 4.38 to 4.44 us, **1.30x** (3 runs of 100,000 round trips; p99 4.89 to 4.92 us) | **pass** |
| One-direction bandwidth | `ib_write_bw` 105.97 to 106.06 Gb/s (3 runs) | not a TASK-15b target | 79.5 to 103.1 Gb/s, median 98.3 (93%; 5 runs of 200,000 packets, 256 slots) | |

As on loopback (TASK-07), our latency includes sealing and checking each packet and the copy into the send buffer, which `perftest` does not do. The extra over `perftest` is about 1.0 us here and 1.2 us on loopback.

## 30-minute generation runs

OLMoE-1B-7B, every expert loaded, F32 payload, GPU matmuls, `--generate 64 --minutes 30` (run 2: 40). Each round generates 64 tokens for each of the three golden prompts and must repeat round 1's tokens and logits bit for bit. Every run first passed the prompt check against transformers (worst logit error 4.01e-5) and matched all 192 tokens of `generate()`.

| Run | Setup | Rounds | Tokens/s | Memory after round 1, max | Every round identical | Exit |
|---|---|---|---|---|---|---|
| 1, two nodes | rank 0 `gx10-02`, rank 1 `gx10-03` | 508 (30.0 min) | **54.1** (rounds 50.2 to 55.6, one at 17.6) | 4,147 MB, +0.00% | **no**: round 164 differed (see below) | rank 0 exit 1 (soak check), rank 1 exit 0 |
| 1, one node | both ranks on `gx10-02` (RDMA loopback) | 193 (30.1 min) | 20.5 (19.8 to 20.6) | 4,147 MB, +0.00% | yes | 0 |
| 1, one node | both ranks on `gx10-03` (RDMA loopback), at the same time | 189 (30.0 min) | 20.1 (19.7 to 20.2) | 4,148 MB, +0.00% | yes | 0 |
| 2, two nodes | as run 1, with the new diagnostics | 681 (40.0 min) | **54.4** (51.1 to 56.1) | 4,187 MB, +0.00% | **yes** (0 rounds differ); both ranks' weights unchanged | 0 and 0 |

**Scaling:** two nodes generate 2.7 times as fast as one (54.4 against 20.5 tokens/s on the idle `gx10-02`, 20.1 on `gx10-03`). On one node the two processes share a GPU and a CPU, so this is mostly the second GPU, not the link; a single node's RDMA loopback and the cable both take a few microseconds per packet, against about 18 ms per token.

GPU utilization (sampled every 5 minutes in run 1, every minute in run 2) was 47 to 62% on `gx10-02` and 27 to 51% on `gx10-03` with two nodes (after the first sample, taken before the run started), and 85 to 93% on each with one node. Rank 1's peak resident set was 3.46 GB, rank 0's 4.05 GB.

## Open issue: round 164 of run 1

In run 1, rounds 1 to 163 matched round 1 exactly. Round 164 took about 11 seconds instead of 3.5 (17.6 tokens/s) and its tokens or logits differed from round 1's. The soak then kept marking every later round as different, because its flag never reset, so whether rounds 165 to 508 differed is not known; they ran at 53 to 54 tokens/s, slightly below the 54 to 55 before. Nothing was logged: no GPU fallback (`gpu.zig` falls back to the CPU and says so if a GPU matmul fails), no RDMA error, and nothing in either machine's kernel log in that window.

What was ruled out by reading the code: the order rank 0 adds expert results in (fixed, whatever order they arrive in); the GPU kernel (one warp reduction per output, no atomics; copies and launch on one stream); the transport (every packet's slot, generation, and CRC32C are checked on arrival, a send buffer is reused only after its send completes, and both sides decode a packet before releasing its slot).

The soak now reports every round that differs, with the prompt, the first differing token, and the largest logit difference, and with `--minutes` each rank checks at exit that its weights in memory still hash as they did at startup. Run 2, with those checks, went 681 rounds (1.3 times run 1) with no stall, no difference, and unchanged weights on both ranks, so the event did not recur. One earlier run (TASK-11, both ranks on `gx10-03`, rank 1 still on the CPU) also saw one round differ once, so it is probably not specific to two nodes. Until it is caught again with the new output, its cause is unknown.

## Acceptance (TASK-15b)

| Criterion | Result |
|---|---|
| A 2 or 3 node run passes the TASK-07 suite across nodes | **pass**: 2 nodes, 20 of 20 dispatch checks and 20 of 20 MoE checks |
| Single-packet p50 latency within 1.5x of node-to-node `ib_write_lat` | **pass**: 1.30x (4.38 to 4.44 us against 3.34 to 3.41) |
| 30-minute generation run archived under `bench/rc/` with scaling against the single-node soak | **pass**: run 2, 40 minutes at 54.4 tokens/s, 2.7x the single-node 20.5; run 1, its 30-minute repeat, is archived with the open issue above |
