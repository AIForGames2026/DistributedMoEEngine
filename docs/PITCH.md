# TensorFold-Swarm: Technical Pitch

## The problem

The strongest open models are sparse Mixture-of-Experts (MoE) networks with hundreds of billions of parameters. Each token uses only a few experts per layer, but every expert still has to live somewhere in fast memory. A single desktop AI box, like an NVIDIA DGX Spark with 128 GB of memory, cannot hold a frontier MoE on its own, and the usual multi-GPU serving stacks assume datacenter hardware, Python runtimes, and collective libraries built for synchronous training.

## The idea

**Weights stay put, data moves.** Split the experts across several desktop nodes and send each token's hidden state (a few kilobytes) to the node that owns the expert it needs, instead of moving gigabytes of weights.

Three engineering moves make that fast enough to be interactive:

1. **Zero-copy RDMA dispatch.** Token packets travel over RoCE v2 straight into the destination node's memory, with no kernel network stack and no copies. Measured on our hardware: **3.26 µs** for one 7,552-byte token packet ([probe](../bench/platform/gx10-03-2026-10-26.md)).
2. **Lookahead routing.** A small predictor guesses which experts the next layer or two will need, so network transfers and NVMe reads start early and hide behind compute that is already running. The model's own router still decides; predictions only drive prefetching, so output stays exact.
3. **Fine-grained expert streaming.** Experts that do not fit in memory are read from NVMe with `io_uring` direct I/O, one expert at a time, before they are needed.

## Why this hardware

| | DGX Spark (GB10) |
|---|---|
| Memory | 128 GB unified LPDDR5X, shared by the Arm CPU and Blackwell GPU |
| Network | ConnectX-7 at 200 Gb/s; our three Sparks are cabled in a full mesh |
| Compute | Blackwell GPU with native FP4 tensor cores (NVFP4) |
| Storage | Local NVMe for expert streaming and KV cache offload |

Unified memory means the network card, the CPU, and the GPU all touch the same buffers, which is what makes zero-copy dispatch practical.

## What you can see

The project ships an in-browser **WebGL inspector** that shows tokens being routed to experts across the cluster in real time: dispatch, compute, and combine, with prefetch hits and misses colored per expert. The PoC drives it with a simulation calibrated to measured hardware numbers; by the Release Candidate it replays traces recorded from the real engine. Nothing to install: it runs on GitHub Pages.

## Deliverables by December 6

- **Must:** a real MoE model running split across two engine processes (RDMA on one Spark, TCP between the desktop and a Spark), loaded with `io_uring`, soak-tested for an hour, and replayed in the public WebGL inspector, plus a benchmark report.
- **Should:** the lookahead router with exact miss handling, a swarm tracker, and capacity guardrails.
- **Could:** NVMe KV cache offload, compressed expert streaming, a true multi-Spark RDMA run, and live telemetry.

Full scope, schedule, risks, and verified hardware limits: [Plan.md](../Plan.md).

## Top risks

1. **Zig is new to the team.** Tested by the TASK-01 toolchain gate; fallback is a C11 core with the same C ABI.
2. **One shared node with about 11 GB free.** Tested by the TASK-05 probe; fallback is small models, a memory cap per process, and TCP or Soft-RoCE transports.
3. **NVMe streaming too slow.** Tested by TASK-03 against a `fio` baseline; fallback is keeping all experts resident.
