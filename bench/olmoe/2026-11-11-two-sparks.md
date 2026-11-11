# OLMoE-1B-7B on two Sparks (2026-11-11)

First run with the two ranks on two machines: rank 0 (attention, router, LM head, experts 0 to 31 of each layer) on `gx10-02`, rank 1 (experts 32 to 63) on `gx10-03`, over RDMA on the direct cable between them (`rocep1s0f0` on `gx10-02`, `rocep1s0f1` on `gx10-03`, 192.168.197.0/24, default GID). Same build as TASK-14 (commit `f07b91a`, ReleaseFast, `-Drdma=true -Dgpu=true`), built on each machine, with the same model file (SHA-256 checked on both) read from each machine's own drive.

```
gx10-03$ olmoe rdma-expert MODEL --port P --dev rocep1s0f1 --gpu --payload f32 --generate 64 --minutes 2 [cache options]
gx10-02$ olmoe rdma-router MODEL --host 192.168.197.13 --port P --dev rocep1s0f0 --golden olmoe_generate.tfref --gpu --payload f32 --generate 64 --minutes 2 [cache options]
```

`gx10-02` was otherwise idle; `gx10-03` still had the co-tenant's vLLM job (about 106 of 121 GB). This is a first look for TASK-15b, not its acceptance run: the TASK-07 suite across nodes, latency against `ib_write_lat`, and a 30-minute run are still to do.

## Results

Every run passed the prompt check against transformers (worst logit error 4.01e-5) and matched all 192 tokens of `generate()`. With the predictor on, it was bit-identical to the predictor off. Every 2-minute soak round matched its reference (the predictor-off round when hints were on), and resident memory stayed flat (+0.00%).

Tokens/s on rank 0, against the same setup with both ranks on `gx10-03` (TASK-14's sweep, `2026-11-10-miss-handling.md`):

| Slots per rank | Hints | Loaders | Both on `gx10-03` | Two Sparks |
|---|---|---|---|---|
| all (no cache) | | | 20.4 | **55.6** |
| 128 (25%) | off | 2 | | 14.1 |
| 128 | off | 4 | 10.2 | 14.9 |
| 128 | top 8 | 2 | 9.8 | 17.0 |
| 128 | top 8 | 4 | 9.2 | 15.3 |
| 128 | top 8 + 2 | 2 | 9.8 | **17.3** |
| 256 (50%) | off | 2 | | 22.4 |
| 256 | off | 4 | 14.5 | 23.4 |
| 256 | top 8 | 2 | 14.6 | **26.3** |
| 256 | top 8 | 4 | 14.2 | 25.3 |
| 256 | top 8 + 2 | 2 | 13.9 | 25.5 |

Rank 0's cache, two Sparks:

| Slots | Hints | Loaders | Already loaded | Hint in time | Hint still loading | Missed | Load time | Stale hints dropped |
|---|---|---|---|---|---|---|---|---|
| 128 | off | 2 | 42.7% | | | 56.9% | 2.16 ms | |
| 128 | top 8 + 2 | 2 | 44.1% | 15.0% | 15.7% | 25.2% | 2.22 ms | 34,112 |
| 256 | off | 2 | 71.4% | | | 27.5% | 2.09 ms | |
| 256 | top 8 | 2 | 72.6% | 11.7% | 5.3% | 10.4% | 2.20 ms | 8,453 |

What this shows:

- **With every expert loaded, two machines run 2.7 times as fast** (55.6 against 20.4 tokens/s). Each rank has a GPU and CPU to itself, and rank 0's machine is idle; these runs do not separate those causes from the link itself.
- **Each rank now reads from its own drive.** An expert takes 2.1 to 2.3 ms to read with two loaders, against 3.2 to 3.7 ms when both ranks shared `gx10-03`'s drive.
- **Hints now pay off.** Against no hints with the same two loaders, they add 23% at 128 slots (14.1 to 17.3) and 17% at 256 (22.4 to 26.3). With both ranks on one drive they added at most 12%.
- **Two loaders beat four once hints are on**, at both sizes, so the default stays at 2.
- The cache is still far from the all-loaded speed: at 256 slots rank 0 spent 50 of 120 s with nothing to run but loads.

## Not yet done (TASK-15b)

- The TASK-07 dispatch suite across the two nodes, and single-packet latency against node-to-node `ib_write_lat`.
- A 30-minute generation run archived under `bench/rc/`, with scaling against TASK-15's single-node soak.
- A third node: `gx10-01` is running another job.
