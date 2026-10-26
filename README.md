# TensorFold-Swarm (Distributed MoE Inference Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
<!-- Enable once CI and Pages are live:
[![CI](https://github.com/AIForGames2026/DistributedMoEEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/AIForGames2026/DistributedMoEEngine/actions)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-blue)](https://aiforgames2026.github.io/DistributedMoEEngine/)
-->

> **Status:** Pre-PoC (planning baseline Oct 26, 2026). See [Plan.md](Plan.md) for the roadmap and workboard.
> **Repository:** https://github.com/AIForGames2026/DistributedMoEEngine

## Pitch

TensorFold-Swarm turns a cluster of NVIDIA GB10 desktop nodes into one inference engine for frontier-scale sparse Mixture-of-Experts models. Instead of copying weights around, it pins each expert to a node and moves only the tokens: a Zig/C engine core with no Python runtime sends token hidden states node to node over zero-copy RoCE v2 RDMA, straight into the destination's LPDDR5X memory. A lightweight lookahead router predicts which experts the next layers will need, so network transfers and `io_uring` NVMe expert reads start before the layer that needs them, hidden behind the compute already running. Everything the swarm does can be watched live in an in-browser WebGL inspector, with nothing to install.

## Links

- [Complete Game/Engine Design Document](docs/GDD.md) *(placeholder)*
- [Technical Pitch Deck](docs/PITCH.md) *(placeholder)*
- [Project Plan & Roadmap](Plan.md)
- [Wire Format Spec](docs/wire-format.md) *(lands with TASK-06)*
- [Telemetry Schema](docs/telemetry-schema.md) *(lands with TASK-04)*
- [Architecture Whitepaper](docs/WHITEPAPER.pdf) *(lands with TASK-19)*
- [Live WebGL Demo](https://aiforgames2026.github.io/DistributedMoEEngine/) *(lands with TASK-18)*

## Team Roles & Members

This is a one-person project built with an AI pair programmer. Luke holds all four roles; the role split shows which hat each part of the work needs.

| Role | Member | Owns |
|---|---|---|
| **Lead Architect** | Luke Hinojosa | Engine core, scheduling loop, gating, lookahead router, miss handling, whitepaper |
| **Network/RDMA Engineer** | Luke Hinojosa | Transport layer, RoCE v2 verbs, swarm tracker integration, multi-node testing |
| **CUDA/Kernel Engineer** | Luke Hinojosa | FP4 GEMM kernels, `io_uring` storage path, expert loader, KV tiering, decompression, platform probe |
| **WebGL Frontend Developer** | Luke Hinojosa | WebGL inspector, telemetry replay, GitHub Pages showcase |
| *AI pair programmer* | Claude (Anthropic) | Code drafts, tests, reviews, and documentation, all reviewed and committed by Luke |

## Target Platform

| Component | Target |
|---|---|
| **Compute node** | NVIDIA DGX Spark (GB10 Grace Blackwell): 20-core Arm Grace CPU, Blackwell GPU (compute capability 12.1, `sm_121`), 128 GB unified LPDDR5X (~273 GB/s). The project starts on one shared Spark, with up to three as they free up. |
| **Dev workstation** | RTX 5070 Ti (16 GB, compute capability 12.0), i7-12700K, 64 GB DDR4, running WSL2; also serves as a second node in cross-machine TCP runs |
| **Interconnect** | NVIDIA ConnectX-7, RoCE v2 (200 Gbps per DGX Spark spec; link speed verified per node by `tools/probe`) |
| **Storage** | PCIe Gen 5 x4 NVMe, read through `io_uring` + `O_DIRECT` |
| **Weight formats** | NVFP4, MXFP4, Q8_0 (GGUF) |
| **OS** | Linux (aarch64); the engine runs in user space with no Python runtime |
| **Web inspector** | Any current browser with WebGL 2.0 and WebAssembly (Chrome, Firefox, Safari) |

## Architecture at a Glance

| Layer | Subsystem | Technology |
|---|---|---|
| 4 | Swarm orchestration (node registry, health, expert ownership) | Lumabri / Colibri pure-C tracker daemon |
| 3 | Scheduling, token batching, lookahead loop, arena allocation | TensorFold engine core (Zig) |
| 2 | Token transport and weight streaming | libibverbs (RoCE v2), liburing, nvCOMP |
| 1 | Expert execution kernels | Blackwell FP4 Tensor Core kernels (CUDA / PTX) |

Four bottlenecks, four answers:

1. **All-to-all network choke**: one-sided RDMA WRITE-with-immediate into pre-registered slot rings, with receiver-granted credits.
2. **Routing dependency stalls**: a lookahead router predicts expert sets 1 to 2 layers ahead; the native router stays authoritative, so predictions affect only prefetching.
3. **PCIe streaming bottleneck**: fine-grained expert reads via `io_uring` fixed buffers, with optional lossless compression only where measurements show a win.
4. **Memory contention**: tiered KV cache keeping attention sink and recent tokens in LPDDR5X and offloading middle context to NVMe.

Design details and verified hardware limits are in [Plan.md, Section 6](Plan.md#6-research-verified-constraints--risk-register).

## Web Delivery & Safety

- The public demo is **100% WebGL 2.0 + WebAssembly, in the browser sandbox**.
- **Showcase visitors never download** `.exe`, `.dmg`, `.pkg`, `.msi`, or install scripts. CI rejects any such file in the Pages bundle.
- The hosted demo replays recorded traces. Live mode connects only to an operator-supplied `wss://` endpoint and is never enabled by default.
- Engine builds on GitHub Releases are for operators running their own GB10 hardware, not for web visitors.

## Repository Layout (planned)

```
build.zig            Zig build entry (run, test, bench)
include/             Public C headers (tensorfold.h)
src/                 Zig engine core: scheduler, arena, gating, lookahead
csrc/
  io/                io_uring O_DIRECT reader
  net/               libibverbs RoCE v2 transport
kernels/             CUDA / PTX FP4 expert kernels
tools/
  probe/             GB10 platform capability probe
  ref/               PyTorch reference harness (testing only, not linked)
web/                 WebGL inspector + Wasm telemetry decoder
bench/               Benchmark scripts and recorded results
docs/                GDD, pitch deck, wire format, telemetry schema, whitepaper
Research/            Source research reports
```

## Getting Started (planned, finalized in TASK-20)

Development happens in WSL2 (Ubuntu 24.04) on Windows or any Linux machine; published benchmarks run only on DGX Spark (GB10) nodes.

### Machine access

Lab machines are reached over Tailscale with SSH keys, never passwords stored in the repo. One-time setup from Git Bash or WSL2:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/spark_ed25519 -C "tensorfold-swarm"
```

```bash
ssh-copy-id -i ~/.ssh/spark_ed25519.pub admin@gx10-03
```

Then add a host entry to `~/.ssh/config`:

```
Host gx10-03
    User admin
    IdentityFile ~/.ssh/spark_ed25519
```

Prerequisites: Linux aarch64 (or x86_64 for CPU-only development), Zig (version pinned in `.zig-version`), CUDA Toolkit with `sm_121` support, `rdma-core`, `liburing`.

```bash
zig build test
```

```bash
zig build run
```

The web inspector builds separately:

```bash
cd web && npm install && npm run build
```

## Milestones

| Milestone | Date | Focus |
|---|---|---|
| PoC | Nov 1, 2026 | Engine loop, mock dispatch, io_uring prototype, platform probe, WebGL mockup |
| Alpha | Nov 8, 2026 | RDMA transport (single-node loopback first), expert loader, first two-process forward pass |
| Beta | Nov 22, 2026 | Lookahead router, tiered KV cache, compressed streaming, miss handling |
| Release Candidate | Nov 29, 2026 | 1-hour soak on one Spark, cross-machine run, multi-Spark stretch, telemetry replay, capacity guardrails |
| Final | Dec 6, 2026 | Public release, docs freeze, benchmark whitepaper |

## Planned Publish Locations

- **GitHub Pages**: in-browser WebGL cluster inspector and demo playground.
- **GitHub Releases**: tagged source releases, operator builds for GB10, and benchmark data bundles.

## Contributing

Work is tracked on the project board with columns `Backlog`, `To Do`, `In Progress`, `Review`, `Done`. Each task carries an area label (`code`, `architecture`, `benchmarking`, `webgl`, `docs`) and a priority label (`must`, `should`, `could`). See [Plan.md, Section 4](Plan.md#4-workboard-snapshot) for board rules.

## Acknowledgements & References

Builds on ideas from TensorFold, Colibri / Lumabri, Strata, Director, SpecPrefetch, ProMoE, fabric-lib, and DeepEP. Full citations are in [`Research/`](Research/).

## License

TensorFold-Swarm is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (`GPL-3.0-or-later`). See [LICENSE](LICENSE).

Before the first binary release, a GPLv3 section 7 additional permission will be added for linking with NVIDIA's proprietary CUDA and nvCOMP libraries (tracked as risk R8 in [Plan.md](Plan.md)).
