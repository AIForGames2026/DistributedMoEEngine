# Project overview

Background on the project: what the PoC proved, who owns what, the target hardware, and how the work is organized. For the plan and task list, see [Plan.md](../Plan.md).

## Proof of Concept

PoC demo video: [watch (4:20)](https://aiforgames2026.github.io/DistributedMoEEngine/media/poc-demo.mp4).

[![TensorFold-Swarm inspector](images/inspector.png)](https://aiforgames2026.github.io/DistributedMoEEngine/)

**Core mechanic: routing tokens to experts across machines.** The WebGL inspector ([open it live](https://aiforgames2026.github.io/DistributedMoEEngine/)) animates a Mixture-of-Experts model (OLMoE-1B-7B shape: 16 layers, 64 experts, top-8) running across the real lab topology:

- Each batch of tokens goes through a top-k gate, layer by layer. Packets fly to whichever node owns each chosen expert (**dispatch**), the experts light up (**compute**), and the results fly back (**combine**).
- Expert colors show what the lookahead router achieved: already in memory, prefetched in time, or a miss that stalls on NVMe.
- Everything on screen is played from a **telemetry trace** ([schema v0](telemetry-schema.md)), the same format the engine will record in TASK-16. Pick a trace: one generated in the browser from simulated routing (switch between the three-Spark cluster and today's Spark-plus-desktop setup, and adjust top-k and predictor recall), a 10,000-packet stress test, or a `.json` trace file of your own.
- Timing in synthetic traces uses **measured** numbers: 3.26 µs RDMA per 7,552-byte packet, 7.2 GB/s NVMe reads, and the Tailscale relay's 33 ms round trip ([`bench/`](../bench/)). The panel lists every input and its source.

**The PoC check:** `dispatch --moe` splits a real (tiny, seeded) MoE across two engine processes: one runs the router, the other runs every expert, and they talk only through 7,552-byte wire-format packets. Output logits match PyTorch to 7.2e-7 with identical routing and top-1 tokens, over shared memory, TCP loopback, and desktop to Spark. See [POC.md](../POC.md).

Engine side: the C ABI and reference GEMM kernels, a token transport (shared memory, TCP verified desktop to Spark, and RoCE v2 RDMA at 102 Gb/s on `gx10-03`), the io_uring expert reader (7.2 GB/s, matching `fio`), an expert loader for GGUF and NVFP4 checkpoints (owned experts only, 7.0 GB/s on OLMoE-1B-7B), an OLMoE-1B-7B forward pass split across two processes (over RDMA, TCP, or shared memory, on CPU or CUDA) that matches Hugging Face transformers, and a CPU MoE reference that matches PyTorch golden outputs. CI runs the unit tests and PoC checks on Linux and Windows, builds the RDMA transport, and cross-compiles for the Sparks (aarch64).

## Links

- [Design Document & Plan](../Plan.md): scope, tasks, milestones, risks, and verified hardware constraints
- [Test plan for Testing Session 1](../TestPlan.md)
- [Technical Pitch](PITCH.md)
- [Project Board](https://github.com/orgs/AIForGames2026/projects/3) and [Milestones](https://github.com/AIForGames2026/DistributedMoEEngine/milestones)
- [Platform probe: gx10-03](../bench/platform/gx10-03-2026-10-26.md)
- [Research reports](../Research/)

## Team Roles & Members

This is a one-person project built with an AI pair programmer. Luke holds all four roles and owns every task on the board; the role split shows which hat each part of the work needs.

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
| **Compute node** | NVIDIA DGX Spark (GB10 Grace Blackwell): 20-core Arm Grace CPU, Blackwell GPU (compute capability 12.1, `sm_121`), 128 GB unified LPDDR5X (~273 GB/s). Results so far come from one shared Spark; the design targets up to three. |
| **Dev workstation** | RTX 5070 Ti (16 GB, compute capability 12.0), i7-12700K, 64 GB DDR4, running WSL2; also serves as a second node in cross-machine TCP runs |
| **Interconnect** | NVIDIA ConnectX-7, RoCE v2, 200 Gb/s; the three Sparks are direct-cabled in a full mesh |
| **Storage** | PCIe Gen 4 x4 NVMe on `gx10-03`, read through `io_uring` + `O_DIRECT` |
| **Weight formats** | NVFP4, MXFP4, Q8_0 (GGUF) |
| **OS** | Linux (aarch64); the engine runs in user space with no Python runtime |
| **Web inspector** | Any current browser with WebGL 2.0 (Chrome, Firefox, Edge, Safari) |

## Planned Publish Locations

- **GitHub Pages** (primary, and the Dec 6 host): the in-browser WebGL inspector, deployed by CI when `web/` changes on `main`. After v1, this page is planned to grow into an operator dashboard for nodes, the router, models, and the API endpoint harnesses call, with the inspector as one feature ([TASK-25](../Plan.md#task-25-operator-dashboard)).
- **GitHub Releases**: tagged source releases (`v0.1.0-poc` to `v1.0.0`), operator builds for GB10, and benchmark data bundles.

## Architecture at a Glance

| Layer | Subsystem | Technology |
|---|---|---|
| 4 | Swarm orchestration (node registry, health, expert ownership) | TensorFold tracker (Zig), modeled on Lumabri's index-only tracker |
| 3 | Scheduling, token batching, lookahead loop, arena allocation | TensorFold engine core (Zig) |
| 2 | Token transport and weight streaming | libibverbs (RoCE v2), liburing, nvCOMP |
| 1 | Expert execution kernels | Blackwell FP4 Tensor Core kernels (CUDA / PTX) |

Four bottlenecks, four answers:

1. **All-to-all network choke**: one-sided RDMA WRITE-with-immediate into pre-registered slot rings, with receiver-granted credits.
2. **Routing dependency stalls**: a lookahead router predicts expert sets 1 to 2 layers ahead; the native router stays authoritative, so predictions affect only prefetching.
3. **PCIe streaming bottleneck**: fine-grained expert reads via `io_uring` fixed buffers, with optional lossless compression only where measurements show a win.
4. **Memory contention**: tiered KV cache keeping attention sink and recent tokens in LPDDR5X and offloading middle context to NVMe.

Design details and verified hardware limits are in [Plan.md, Section 7](../Plan.md#7-risks-research-verified-constraints--fallbacks).

## Web Delivery & Safety

- The demo is **100% WebGL 2.0 + JavaScript in the browser sandbox**, served as static files from GitHub Pages.
- **Visitors never download** `.exe`, `.dmg`, `.pkg`, `.msi`, or install scripts. The Pages workflow fails the deploy if any such file appears in the bundle.
- The page sets a Content-Security-Policy and loads no third-party scripts.
- The hosted demo uses simulated or recorded data only. Live telemetry (TASK-16b) would connect only to an operator-supplied `wss://` endpoint and is never enabled by default.

## Repository Layout

```
build.zig            Zig build (run, test); .zig-version pins Zig 0.17.0
include/             Public C ABI (tensorfold.h, tf_wire.h)
csrc/                C sources behind the ABI (io_uring reader, CRC32C; verbs code lands here)
src/                 Zig engine core (entry point, arena, wire format, transport, reference GEMM)
tests/               Plain C tests of the public ABI; golden/ holds PyTorch reference outputs
tools/probe/         Platform probe (probe.py writes bench/platform/<host>-<date>.json)
tools/ref/           PyTorch reference harness for the golden files (testing only)
tools/setup/         Zig installers (Linux, Windows) and WSL2 package setup
web/                 WebGL inspector and guided mission (Vite + Three.js), trace generator
bench/platform/      Hardware probe results
docs/                Pitch, wire format, telemetry schema, images, and (later) whitepaper
Research/            Source research reports
.github/workflows/   CI (engine tests, web build) and GitHub Pages deploy
```

## Workflow

Work is tracked on the [project board](https://github.com/orgs/AIForGames2026/projects/3) with columns `Backlog`, `To Do`, `In Progress`, `Review`, `Done`. Every task has a priority label (`must`, `should`, `could`), an area label (`code`, `architecture`, `benchmarking`, `webgl`, `docs`), a milestone, and, if it is in a cut line, a `cut-1` or `cut-2` tag. See [Plan.md, Sections 4 and 5](../Plan.md#4-scope-tiers--cut-lines) for scope tiers, cut lines, and board rules.

## Acknowledgements & References

Builds on ideas from TensorFold, Colibri / Lumabri, Strata, Director, SpecPrefetch, ProMoE, fabric-lib, and DeepEP. Full citations are in [`Research/`](../Research/).

## Licensing note

Before the first binary release, a GPLv3 section 7 additional permission will be added for linking with NVIDIA's proprietary CUDA and nvCOMP libraries (tracked as risk R8 in [Plan.md](../Plan.md)).
