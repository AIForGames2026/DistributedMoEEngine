# Project Plan & Roadmap: TensorFold-Swarm

> Plan baseline: Monday, Oct 26, 2026. Milestone dates fall on Sundays; each milestone closes at end of day on its date.
> Repository: https://github.com/AIForGames2026/DistributedMoEEngine
> Team: one engineer (Luke Hinojosa) working with an AI pair programmer (Claude). All tasks are owned by Luke.
> Board: https://github.com/orgs/AIForGames2026/projects/3 (issues #1 to #22 mirror the tasks below). Live demo: https://aiforgames2026.github.io/DistributedMoEEngine/
> Sources: `Research/GB10 Distributed MoE Engine.pdf` (system blueprint) and `Research/NVIDIA Blackwell Hardware Decompression Engine.pdf` (platform verification). Where the two disagree, the verification report wins. See Section 7.

### Hardware

| Machine | What it is | Availability | Role in this project |
|---|---|---|---|
| `gx10-03` | NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, ConnectX-7) | **Now**, shared with another student (about 11 GB of memory free while their vLLM job runs) | Primary engine node and all published benchmarks |
| `gx10-01`, `gx10-02` | NVIDIA DGX Spark (GB10) | **Busy** running GLM 5.3; may free up later | Multi-node RDMA if and when available (TASK-15b) |
| Desktop | Windows 11 workstation: RTX 5070 Ti (16 GB, Blackwell, compute capability 12.0), i7-12700K, 64 GB DDR4 | **Always** | Development in WSL2, local CUDA kernel work, and the second node in cross-machine TCP runs |

**Probe results for `gx10-03` (Oct 26, read-only):**
- Ubuntu 24.04.5, kernel 7.0 (NVIDIA), driver 580, CUDA 13.0; GPU reports compute capability 12.1.
- **Memory is the binding constraint:** 121 GB total, only about 11 GB available, swap in use. A vLLM engine (root, under k3s) holds about 100 GB of GPU memory.
- **ConnectX-7:** all four RoCE interfaces are `ACTIVE` at 200 Gb/s with MTU 9000, and the three Sparks are **direct-cabled in a full mesh**: one QSFP port goes to `gx10-01`, the other to `gx10-02`. Node-to-node RDMA needs only free machines, not new cables.
- **NVMe:** a 1 TB drive on **PCIe Gen 4 x4** (16 GT/s), not Gen 5, so roughly 7 GB/s is the ceiling; 230 GB of disk is free; root filesystem is ext4; io_uring is enabled.
- **RDMA loopback works:** `ib_write_lat` gives 1.37 us typical at 64 bytes and 3.26 us for a 7,552-byte packet; `ib_write_bw` gives about 106 Gb/s on a single device. Each 200 Gb/s QSFP port shows up as two devices, so TASK-07 should stripe traffic across both devices per peer to approach full rate. Details: `bench/platform/gx10-03-2026-10-26.md`.
- **Installed Oct 26** in `~/lukehinojosa/tools/`: Zig 0.17.0 (checksum verified), liburing 2.15, fio 3.43.
- **Tools:** `gcc`, `cmake`, `python3`, verbs headers, `ib_write_lat` / `perftest`, and `ethtool` are present. `zig`, `fio`, `clang`, and the liburing headers are missing; all of them can be built without `sudo`. Because `admin` is a shared account, everything for this project lives in one folder, `~/lukehinojosa/` (tools, source, scratch files, and an `env.sh` sourced manually); `~/.bashrc` and other shared files are never modified.

Machine time is informal and flexible: there is no booking schedule. Because `gx10-03` is shared, check with the other student before long runs or anything that uses most of the node's memory or GPU, and record co-tenant load next to every benchmark.

## 1. Executive Summary & Delivery Requirements
- **Core Delivery Target**: A user-space C/Zig engine core (no Python runtime) for distributed expert-parallel MoE inference on NVIDIA GB10 (DGX Spark) nodes, plus an interactive in-browser WebGL Token Stream Visualizer.
- **Safety Standard**: 100% WebGL / WebAssembly client sandbox for web delivery, with zero downloaded `.exe` / `.dmg` binaries or script installers for showcase visitors. Engine source and operator builds ship through GitHub Releases for people running their own hardware; the public demo never asks a visitor to download or run anything.
- **Architecture Principle**: "Weights stay put, data moves." Experts are sharded across engine processes (one per node, or several per node while only one node is free); token hidden states travel between them over RoCE v2 RDMA where the hardware allows, otherwise over TCP; non-resident experts stream from NVMe ahead of need, guided by a lookahead router.
- **Measurement Principle**: Every published number comes from a recorded run on a named machine with its full configuration. Figures from the blueprint that the verification report flags as unestablished (Section 7) are treated as hypotheses to measure, never as facts to quote.
- **Development Environment**: Code is written and unit-tested in WSL2 (Ubuntu 24.04) on the desktop, where the RTX 5070 Ti also supports CUDA work. It is then built and benchmarked natively on `gx10-03` over Tailscale SSH with key-based login. No headline number is taken from WSL2.

---

## 2. Master Milestone Schedule

| Milestone | Target Date | Core Focus | Git Tag |
|---|---|---|---|
| **PoC** | Nov 1, 2026 | Zig engine loop, mock token dispatch, io_uring prototype, platform probe, wire format, & WebGL mockup | `v0.1.0-poc` |
| **Alpha** | Nov 8, 2026 | RDMA transport (single-node loopback first), quantized expert loader, & first two-process forward pass | `v0.2.0-alpha` |
| **Beta** | Nov 22, 2026 | Predictive lookahead router, miss handling, swarm tracker, & (stretch) KV tiering and compressed streaming | `v0.3.0-beta` |
| **Release Candidate** | Nov 29, 2026 | 1-hour soak on `gx10-03`, cross-machine run with the desktop, recorded-trace visualizer, & capacity guardrails | `v0.9.0-rc1` |
| **Final** | Dec 6, 2026 | Public release, docs freeze, and benchmark report | `v1.0.0` |

---

## 3. Workboard Task Matrix

Every task carries one priority label (`must`, `should`, `could`), one area label, and, if it is in a cut line, a `cut-1` or `cut-2` tag (Section 4).

### Milestone 1: PoC (Target: Nov 1): *Immediate Monday Actionable*

#### [TASK-01] Zig Engine Core Skeleton & Toolchain Gate
- **Milestone**: PoC (Nov 1)
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Set up the WSL2 and `gx10-03` toolchains, then initialize the C/Zig engine skeleton with a page-aligned arena allocator, a C-ABI boundary for C and CUDA components, and CI. Replaces the CLion `CMakeLists.txt` / `main.cpp` stub with `build.zig`. This task is the test for Risk R1 (Section 7).
- **Steps (Monday start)**:
  - [x] On `gx10-03`: Zig 0.17.0, liburing 2.15, and fio 3.43 installed in `~/lukehinojosa/tools/` (Oct 26); `zig cc` builds and runs a liburing test.
  - [x] Install WSL2 Ubuntu 24.04 with the NVIDIA WSL CUDA toolkit; install Zig 0.17.0 (pinned in `.zig-version`), `liburing-dev`, `rdma-core`, `libibverbs-dev`, `fio` (Oct 27: `tools/setup/install-zig.sh` and `tools/setup/wsl2-packages.sh`; CUDA 13.0, liburing 2.5, rdma-core 50.0, fio 3.36).
  - [x] Set up key-based SSH to `gx10-03` (README, "Machine access"); done Oct 26 for all three Sparks.
  - [ ] Work through the Zig ramp-up list in Section 9 (about one day).
  - [x] Create `build.zig` with `run` and `test` steps; compile one C file (`csrc/tf_version.c`) and call it from Zig (Oct 27; `bench` step still to add).
  - [x] **Toolchain gate (Wed Oct 28)**: in WSL2, `zig build` compiles C, links `liburing` and `libcudart`, calls `cudaGetDeviceCount` on the 5070 Ti, and cross-compiles for `aarch64-linux-gnu`; the same build runs natively on `gx10-03`. (**Passed Oct 27, one day early.** `zig build gate` passes in WSL2 (RTX 5070 Ti, 12.0, 70 SMs) and on `gx10-03` (GB10, 12.1, 48 SMs): io_uring NOP reaped, cudaMalloc and cudaMemset ok, no libpython. The aarch64 cross-compile from WSL2 works. Risk R1's fallback is not needed.)
  - [x] Implement `Arena` (2 MiB aligned, huge-page aware) with allocation tracing hooks (Oct 27; `src/arena.zig`).
  - [x] Add a naive and a tiled FP32 GEMM on host memory as the correctness reference kernel (Oct 27; tests pass on `gx10-03`).
  - [x] Expose `include/tensorfold.h` and call it from a C test to prove the ABI (Oct 27; `tests/abi_test.c` runs in `zig build test`).
  - [x] GitHub Actions: `zig build test` on x86_64 Linux plus an aarch64 cross-compile (`.github/workflows/ci.yml`).
  - **Results (Oct 27)**: all acceptance checks pass locally. 1024x1024 tiled GEMM matches naive exactly (0 relative error); `zig build startup` measures process start to first kernel at a median of 0.30 ms on `gx10-03` and 0.93 ms in WSL2 (target < 5 ms); no libpython.
- **Acceptance Criteria**: Done when the toolchain gate passes, `zig build run` executes a 1024x1024 host matmul that matches the naive reference within 1e-5 relative error, the binary links no Python or libpython, process start to first kernel call is under 5 ms (median of 20 runs on `gx10-03`), and CI is green.

#### [TASK-02] Mock All-to-All Token Dispatcher (Loopback + TCP Transport)
- **Milestone**: PoC (Nov 1)
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement a `Transport` interface with a shared-memory loopback backend and a TCP backend that simulate token hidden-state dispatch and combine between 2 engine processes. The interface must match what the RDMA backend needs in TASK-07: pre-registered slot rings, slot credits, and a per-message 32-bit immediate tag. The TCP backend is what links the desktop and `gx10-03`, and is the fallback for Risk R2.
- **Steps (Monday start)**:
  - [x] Define the `Transport` vtable: `post_write_imm`, `post_imm`, `poll_completions`, `slot` (`src/transport.zig`). Credit grants and a goodbye handshake ride on `post_imm`; the backend owns ring registration (Oct 27).
  - [x] Implement the shared-memory backend with two processes and fixed-size slot rings (`src/transport/shm.zig`, memfd + fork).
  - [x] Implement the TCP backend with the same semantics (framing carries the immediate value) (`src/transport/tcp.zig`).
  - [x] Use the v0 packet layout from TASK-06.
  - [x] Add CRC32C per packet and a full 64-bit generation counter in the slot header (CRC uses ARMv8 / SSE4.2 instructions, `csrc/tf_crc32c.c`).
  - **Results (Oct 27)**: all acceptance checks pass. 4,096 + 4,096 packets verified over shared memory (1.8 us per token on `gx10-03`), TCP loopback, and TCP from WSL2 to `gx10-03`; credit-withholding test passes; backend chosen by the `dispatch` mode argument. See `bench/transport/2026-10-27.md`.
- **Acceptance Criteria**: Done when two processes exchange 4,096 token hidden-state packets in each direction (dispatch then combine) over both backends, including TCP between the desktop (WSL2) and `gx10-03`; every packet passes CRC and generation checks; a test that withholds credits proves the sender blocks instead of overwriting an unconsumed slot; and the backend is selected by one build or runtime flag.

#### [TASK-03] `io_uring` O_DIRECT Expert Reader Prototype
- **Milestone**: PoC (Nov 1)
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Create a C wrapper around liburing using `O_DIRECT`, `IORING_SETUP_SQPOLL`, registered files, and registered fixed buffers (`io_uring_prep_read_fixed`) to load dummy 1 GB expert weight files into page-aligned memory. Alignment comes from `statx(STATX_DIOALIGN)`, not hard-coded constants. This task is the test for Risk R3.
- **Steps (Monday start)**:
  - [x] Port the reference fixed-buffer loop from the verification report (Section 4.3) into `csrc/io/uring_reader.c`; unit-test in WSL2 (Oct 27; library API in `include/tf_io.h`, test `zig build io-test`).
  - [x] Generate 1 GB aligned dummy expert files with a deterministic pattern for checksumming (check free disk space on `gx10-03` first; delete test files after runs) (`uring_bench mk`; deleted after the run).
  - [x] On `gx10-03`, record a `fio` O_DIRECT baseline (256 KiB, QD 64) as the ceiling to compare against (7.21 to 7.26 GB/s).
  - [x] Report GB/s plus p50/p95/p99 submit-to-reap latency; sweep block size (128 KiB to 1 MiB) and queue depth (16 to 128).
  - **Results (Oct 27)**: all acceptance checks pass. Reader 7.24 to 7.27 GB/s, about 100% of `fio`; all words verified; page cache +0 MiB. The drive saturates once about 4 MiB is in flight, so the engine default is 256 KiB x QD 16 (p50 0.57 ms). Risk R3 is retired for single-drive reads. See `bench/storage/gx10-03-2026-10-27.md`.
- **Acceptance Criteria**: Done when, on `gx10-03`, the reader sustains at least 90% of the `fio` O_DIRECT baseline (the drive is PCIe Gen 4 x4, so expect roughly 6 to 7 GB/s; build `fio` under `~/lukehinojosa/tools/` since it is not installed), reads verify by checksum, the kernel page cache does not grow during the run (checked via `/proc/meminfo` `Cached` delta or `fincore`), and results are logged with drive model, firmware, kernel, filesystem, and co-tenant load.

#### [TASK-04] WebGL Token Flow Canvas Mockup
- **Milestone**: PoC (Nov 1)
- **Status**: Review
- **Priority**: `must` | **Area**: `webgl`
- **Description**: Build a lightweight WebGL 2.0 canvas (Three.js) displaying node topology and animated token packet routing, driven by a synthetic JSON trace whose schema the engine will emit later (TASK-16).
- **Steps (Monday start)**:
  - [x] Scaffold `web/` as a Vite static site that builds to a folder deployable on GitHub Pages (Oct 27; deployed by `.github/workflows/pages.yml`).
  - [x] Draft `docs/telemetry-schema.md` v0: node, process, expert, layer, token-batch, and timestamp fields (Oct 27; compact positional events `L`, `D`, `X`, `C`, `F`; readers ignore unknown kinds).
  - [x] Render the cluster presets (three Sparks, or `gx10-03` + desktop), per-node expert grids, and instanced token packets moving along links, driven by a simulated top-k router (Oct 27).
  - [x] Add a trace player (play, pause, and speed done; scrubbing waits for recorded traces) (Oct 27; the renderer now consumes only traces: generated in the browser by `web/src/trace.js`, the 10,000-packet stress trace, or a loaded `.json` file; progress bar; `npm test` checks the generator in CI).
  - **Results (Oct 27)**: the stress trace (10,984 drawn packets, up to 493 in flight) plays with 0.9 to 2.0 ms of main-thread work per frame during playback (60 FPS needs under 16.7 ms), on the desktop's RTX 5070 Ti in the Claude browser pane. That pane caps `requestAnimationFrame` at 30 FPS, so the frame rate was confirmed by hand on the desktop: **144 FPS on the stress trace in Firefox and in Edge** (Chromium, standing in for Chrome, which is not installed), shown by the HUD's "Render FPS" and capped by the 144 Hz display. Files load and validate; bad versions are rejected with a message; no horizontal scroll at 375 px.
- **Acceptance Criteria**: Done when the canvas renders a 10,000-packet synthetic trace at 60 FPS in desktop Chrome and Firefox, opens from a static file server with no plugins, installers, or downloads, and a browser without WebGL2 sees a readable fallback message instead of a blank page.

#### [TASK-05] Platform Capability Probe (`gx10-03` + Desktop)
- **Milestone**: PoC (Nov 1)
- **Status**: Review
- **Priority**: `must` | **Area**: `benchmarking`
- **Description**: Run a probe on `gx10-03` and the desktop that records the facts later tasks depend on, so design decisions rest on measured hardware instead of datasheet assumptions (Section 7). This task is the test for Risk R2.
- **Steps (Monday start)**:
  - [x] Query compute capability and SM count (Oct 27: 12.1 and 48 SMs on `gx10-03` via `tools/probe/cuda_probe.c`; 12.0 on the 5070 Ti via `nvidia-smi`).
  - [x] Try to assemble a `tcgen05` PTX snippet for `sm_121a` to confirm whether GB10 has the datacenter Blackwell Tensor Memory path (Oct 27: it does not; see `bench/platform/gx10-03-2026-10-27.md`).
  - [x] Query `CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH` and test a buffer with `CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE` to detect a hardware Decompression Engine (Oct 27: algorithm mask 0, no hardware DE).
  - [x] On `gx10-03`, record `ibdev2netdev` link state and speed for the ConnectX-7 ports (done Oct 26: all four `ACTIVE` at 200 Gb/s, full mesh to `gx10-01` / `gx10-02`).
  - [x] Record the RoCE GID table; run `ib_write_lat` and `ib_write_bw` in loopback (done Oct 26: 3.26 us typical for a 7,552-byte packet, about 106 Gb/s on one device; see `bench/platform/gx10-03-2026-10-26.md`).
  - [x] Record `lspci -vv` NVMe link speed and width, free disk space, total and free memory, and a STREAM-style memory bandwidth test; note what the other student's jobs are using (Oct 27; NVMe link from sysfs, `tools/probe/stream.c`).
  - **Results (Oct 27)**: `tools/probe/probe.py` writes `bench/platform/gx10-03-2026-10-27.json` and `bench/platform/desktop-wsl2-2026-10-27.json`; each has an `answers` block. `gx10-03`: compute 12.1, no TMEM, 4 ports ACTIVE, RDMA loopback 3.17 us (7,552 B), no hardware DE, CPU triad 126 GB/s, NVMe 7.21 GB/s, 12.3 GB available. Desktop: compute 12.0, no TMEM, no hardware DE, triad 22.8 GB/s. Re-baselined targets: TASK-07 p50 at most 4.8 us (1.5 x 3.17 us) and at least 93 Gb/s (90% of loopback `ib_write_bw` at 7,552 B); TASK-13 uses software decompression only; TASK-15's soak model and buffers must fit in about 12 GB, so it uses an OLMoE-1B-7B-class model with a 4 GB cap per process.
- **Acceptance Criteria**: Done when `tools/probe` emits a JSON report per machine committed under `bench/platform/`, and the report answers: (1) compute capability on each machine, (2) whether `tcgen05` / TMEM assembles for `sm_121a`, (3) ConnectX-7 port state and whether `ib_write_lat` runs in loopback on `gx10-03`, (4) whether the hardware DE path exists, and (5) measured memory and NVMe bandwidth plus how much memory is realistically free. TASK-07, TASK-13, and TASK-15 targets are re-baselined from these answers.

#### [TASK-06] Wire Format v0 Spec & Reference Golden Harness
- **Milestone**: PoC (Nov 1)
- **Status**: Review
- **Priority**: `must` | **Area**: `architecture`
- **Description**: Freeze the v0 token packet layout and immediate-data encoding, and build an offline PyTorch reference harness (testing only, never linked into the engine) that produces golden activations and logits for a tiny synthetic MoE. The harness can run on the desktop's 5070 Ti. TASK-02 and TASK-10 depend on it.
- **Steps (Monday start)**:
  - [x] Reconcile the blueprint packet table: its offsets give 208 bytes of scale factors but `fp8_e8m0[224]` needs 224 (7,168 / 32). With 224 scales the packet is 7,440 bytes; pad to 7,552 (59 x 128) or shrink the header. Document the choice (Oct 27: padded to 7,552).
  - [x] Pick one immediate-data layout (the two research documents propose different bit fields) and put the full generation number and length in the slot header, since an 8-bit epoch wraps (Oct 27: slot, kind, gen8, src_rank).
  - [x] Write `docs/wire-format.md` with a byte-offset table and `comptime` / `static_assert` size checks (Oct 27; `src/wire.zig`, `include/tf_wire.h`, cross-checked in `zig build test`).
  - [x] Build `tools/ref/` to export a 2-layer, 8-expert synthetic MoE plus golden inputs and outputs in a simple binary format (Oct 27; `tools/ref/ref_moe.py` writes `tests/golden/moe_tiny.tfref`).
  - **Results (Oct 27)**: all acceptance checks pass. `ref_moe.py --check` regenerates a byte-identical file (PyTorch 2.9.0 CPU, seed 20261101). `src/moe_ref.zig` reruns the model in plain Zig from the embedded golden file and matches PyTorch layer by layer: identical expert routing, values within `torch.allclose` tolerance (rtol 1e-4, atol 1e-5). TASK-10's kernels are checked against the same file.
- **Acceptance Criteria**: Done when the Zig and C structs match `docs/wire-format.md` under compile-time size and offset assertions, and `tools/ref` reproducibly regenerates golden files from a fixed seed.

---

### Milestone 2: Alpha (Target: Nov 8)

#### [TASK-07] RoCE v2 Zero-Copy RDMA Transport
- **Milestone**: Alpha (Nov 8)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement the RDMA backend of the TASK-02 `Transport` with libibverbs Reliable Connection QPs over ConnectX-7, using one-sided `IBV_WR_RDMA_WRITE_WITH_IMM` into pre-registered remote slot rings. Bootstrap QPN, PSN, GID index, remote address, and rkey from a launcher manifest. The CPU publishes receiver-side readiness after it sees the CQE; the immediate value does not increment memory by itself. With one free node, this is developed and accepted as two processes on `gx10-03` talking through its ConnectX-7 (HCA loopback). The Sparks are already direct-cabled in a full mesh at 200 Gb/s, so real two-node runs need only `gx10-01` or `gx10-02` to be free (TASK-15b).
- **Acceptance Criteria**: Done when two processes on `gx10-03` pass the TASK-02 test suite unchanged on the RDMA backend, single-packet one-way latency p50 is within 1.5x of loopback `ib_write_lat` for the same size, sustained bandwidth reaches at least 90% of loopback `ib_write_bw`, and no payload slot is reused before its send completion. If TASK-05 shows RDMA loopback cannot run on `gx10-03`, the same suite passes against Soft-RoCE (`rdma_rxe`) and the Emergency cut in Section 4 applies for performance numbers.

#### [TASK-08] Quantized Expert Loader (GGUF + NVFP4)
- **Milestone**: Alpha (Nov 8)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement a model loader that parses GGUF (Q8_0 and MXFP4 tensor types) and NVFP4 checkpoints (E2M1 values, E4M3 scale per 16 values, FP32 tensor scale) and loads only the experts this engine process owns, reading through the TASK-03 `io_uring` path.
- **Acceptance Criteria**: Done when the engine parses a multi-expert model chunk, loads only this process's experts into memory, validates tensor shapes and checksums against the file metadata, and dequantized values match the reference harness within format tolerance.

#### [TASK-10] Top-k Expert Gating & First Two-Process Forward Pass
- **Milestone**: Alpha (Nov 8)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement the gating router (softmax over top-k logits, starting with k = 1 and generalizing to k = 8) that sends each token to local or remote experts, runs the expert FFN, and combines weighted results. Combine writes each expert contribution to its own slot and reduces locally; overlapping RDMA writes do not sum. Expert placement comes from a static manifest file until TASK-09 lands.
- **Acceptance Criteria**: Done when a token sequence completes a forward pass split across 2 engine processes (on `gx10-03` over RDMA, and between the desktop and `gx10-03` over TCP) for the TASK-06 synthetic MoE and one small public MoE (for example OLMoE-1B-7B), with output logits matching the PyTorch reference within a documented tolerance (target: max abs error at or below 1e-2 in FP8 paths) and identical top-1 next tokens on the golden prompts.

---

### Milestone 3: Beta (Target: Nov 22)

#### [TASK-09] Swarm Node Tracker (Lumabri / Colibri Integration)
- **Milestone**: Beta (Nov 22)
- **Status**: Backlog
- **Priority**: `should` | **Area**: `architecture` | **Tag**: `cut-2`
- **Description**: Replace the static placement manifest with the pure-C Lumabri tracker daemon (TCP/UDP 47300) as the control plane: live process and node registry, health heartbeats, and the expert ownership map the dispatcher reads. Token payloads never pass through it. Check Lumabri's license for GPLv3 compatibility before vendoring.
- **Acceptance Criteria**: Done when a new engine process joins and its expert assignment appears in every peer's ownership map within 2 seconds, a killed process is marked unhealthy within 5 seconds, and the dispatcher refuses to route to an unhealthy process.

#### [TASK-11] Speculative Lookahead Router
- **Milestone**: Beta (Nov 22)
- **Status**: Backlog
- **Priority**: `should` | **Area**: `code` | **Tag**: `cut-2`
- **Description**: Build a lightweight low-rank or cascaded predictor (following SpecPrefetch / Director) that predicts expert sets ahead (H = 1 first, H = 2 target) and issues RDMA pre-arming and NVMe prefetch commands through the Prefetch Command Ring. The native router stays authoritative: predictions only drive transfers, never which experts run. Prefetch keys are `(layer, expert)` pairs. Predictor training can run on the desktop's 5070 Ti.
- **Acceptance Criteria**: Done when prefetches are issued before the target layer starts in at least 95% of steps, measured expert recall at H = 2 with k + 2 candidates is at least 80% on two evaluation datasets (or H = 1 results are reported if H = 2 misses), predictor compute overhead is under 2% of layer time, and outputs are bit-identical with the predictor on versus off.

#### [TASK-14] Misprediction & Miss Handling (Exact Mode)
- **Milestone**: Beta (Nov 22)
- **Status**: Backlog
- **Priority**: `should` | **Area**: `code` | **Tag**: `cut-2`
- **Description**: Handle experts the lookahead missed, without changing model output: cancel obsolete speculative requests, promote true misses to a high-priority queue, run resident experts while missing ones arrive, and fetch from a peer process's memory over RDMA READ when that beats local NVMe. An approximate skip-low-weight mode is out of scope (Section 4).
- **Acceptance Criteria**: Done when, under forced 10% mispredictions for 30 minutes, the engine produces tokens continuously without crash, deadlock, or memory growth, and output is bit-identical to a no-speculation run.

#### [TASK-12] Tiered NVMe KV Cache Offloader
- **Milestone**: Beta (Nov 22)
- **Status**: Backlog
- **Priority**: `could` | **Area**: `code` | **Tag**: `cut-1`
- **Description**: Implement a ring-buffer KV manager that pins attention sink tokens (first 128) and the most recent 8,192 tokens in memory, and offloads cold middle context to NVMe via `io_uring`, paging it back on demand with prefetch.
- **Acceptance Criteria**: Done when context length reaches 128k tokens with under 15% decode throughput degradation versus an all-RAM KV cache at a context length where both fit, and output matches the all-RAM run exactly.

#### [TASK-13] Compressed Expert Streaming (nvCOMP, Hardware DE Only If Present)
- **Milestone**: Beta (Nov 22)
- **Status**: Backlog
- **Priority**: `could` | **Area**: `code` | **Tag**: `cut-1`
- **Description**: Store experts losslessly compressed on NVMe and decompress them into CUDA-accessible memory before the FP4 kernel consumes them. Use the hardware Decompression Engine only if TASK-05 finds it on GB10 (NVIDIA documents it for B200/B300/GB200/GB300, not GB10), selecting `NVCOMP_DECOMPRESS_BACKEND_HARDWARE` so incompatible buffers fail instead of silently falling back. Otherwise use SM-based nvCOMP. The pipeline ends in device memory buffers; there is no documented NVMe to DE to TMEM path.
- **Acceptance Criteria**: Done when the four-way benchmark (NVMe only, GEMM only, NVMe + GEMM, NVMe + decompress + GEMM) is recorded on real NVFP4 expert weights with compression ratio, compressed-input and output GB/s, GEMM slowdown, and end-to-end expert-ready latency, and compression is enabled by default only if it improves expert-ready latency by at least 10% over raw reads.

---

### Milestone 4: Release Candidate (Target: Nov 29)

#### [TASK-15] Soak Test on `gx10-03` + Cross-Machine Run
- **Milestone**: RC (Nov 29)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `benchmarking`
- **Description**: Two runs with a real public MoE whose weights fit in the memory left free on `gx10-03`. With about 11 GB free today, that means a small model such as OLMoE-1B-7B at 4 bits (about 4 GB); a Qwen3-30B-A3B-class model (about 17 GB) only if the co-tenant's vLLM job is paused for the run. **Soak**: two engine processes on `gx10-03`, each owning half the experts, linked by the RDMA backend, generating for 1 hour. **Cross-machine**: the same model split between the desktop and `gx10-03` over TCP, proving the engine runs on two physically separate, mixed machines. Agree on a time with the other student before the soak.
- **Acceptance Criteria**: Done when the soak sustains at least 10 tokens/second aggregate for 1 hour with zero crashes and flat resident memory (under 1% drift after warm-up), the cross-machine run generates correct tokens for 10 minutes (its throughput is recorded but not targeted, since it rides on Tailscale), and both runs' logs (configs, versions, counters, telemetry traces, co-tenant load) are archived under `bench/rc/`.

#### [TASK-15b] Multi-Spark RDMA Run (2 to 3 Nodes)
- **Milestone**: RC (Nov 29)
- **Status**: Backlog
- **Priority**: `could` | **Area**: `benchmarking` | **Tag**: `cut-1`
- **Description**: Only if `gx10-01` or `gx10-02` frees up (cabling is already in place: the three Sparks are direct-cabled in a full mesh): repeat the soak with one engine process per Spark over real node-to-node RDMA.
- **Acceptance Criteria**: Done when a 2 or 3 node run passes the TASK-07 suite across nodes, single-packet p50 latency is within 1.5x of node-to-node `ib_write_lat`, and a 30-minute generation run is archived under `bench/rc/` with scaling against the single-node soak.

#### [TASK-16] Telemetry Recording & WebGL Trace Replay
- **Milestone**: RC (Nov 29)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `webgl`
- **Description**: The engine writes compact telemetry (layer activations, routing decisions, RDMA and NVMe events) to trace files using the TASK-04 schema; the WebGL inspector loads and replays them. These recorded traces are what the public demo ships.
- **Acceptance Criteria**: Done when a TASK-15 run's trace replays in the browser at 60 FPS with correct per-process routing counts (checked against engine counters), and a malformed trace file shows an error message instead of crashing the page.

#### [TASK-16b] Live `wss://` Telemetry & Wasm Frame Decoder
- **Milestone**: RC (Nov 29)
- **Status**: Backlog
- **Priority**: `could` | **Area**: `webgl` | **Tag**: `cut-1`
- **Description**: Stream binary telemetry frames from an operator-run relay over secure WebSockets, decoded in the browser by a Wasm module built from the same Zig schema code. Never enabled by default on the public site.
- **Acceptance Criteria**: Done when the inspector shows a live run over `wss://` with under 500 ms end-to-end delay and the Wasm decoder rejects malformed frames without crashing the page.

#### [TASK-17] Capacity Factor & Memory Thrashing Guardrails
- **Milestone**: RC (Nov 29)
- **Status**: Backlog
- **Priority**: `should` | **Area**: `code` | **Tag**: `cut-2`
- **Description**: Enforce per-expert capacity factors, receiver-granted credits, and a hard memory budget per engine process, so load spikes cannot overflow rings or push a shared machine into swapping. Overflow tokens take the residual path and are counted.
- **Acceptance Criteria**: Done when a synthetic 3x load spike completes with dropped-token counts reported, no memory leak (allocator balance at zero after the run), no pipeline stall longer than 100 ms, the process never exceeds its configured memory budget, and token drop rate is under 1% at nominal load.

---

### Milestone 5: Final Release (Target: Dec 6)

#### [TASK-18] GitHub Pages WebGL Showcase Deployment
- **Milestone**: Final (Dec 6)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `docs`
- **Description**: Host the WebGL cluster inspector at https://aiforgames2026.github.io/DistributedMoEEngine/, shipping recorded traces from the RC runs. Static assets only: HTML, JS, optional Wasm, and trace data.
- **Acceptance Criteria**: Done when the public URL loads the visualizer in current Chrome, Firefox, and Safari without any binary download or install prompt, a CI check fails the deploy if the bundle contains `.exe`, `.dmg`, `.pkg`, `.msi`, or shell installer files, a Content-Security-Policy is set, and initial load is under 5 MB.

#### [TASK-19] Benchmark Report & Architecture Whitepaper
- **Milestone**: Final (Dec 6)
- **Status**: Backlog
- **Priority**: `must` | **Area**: `docs`
- **Description**: Document RDMA latencies, NVMe yields, lookahead recall (if TASK-11 shipped), single-node and cross-machine results, and multi-Spark scaling if TASK-15b ran. Each figure links to the raw run directory that produced it. Cut features are listed as future work, not described as results.
- **Acceptance Criteria**: Done when `docs/WHITEPAPER.pdf` and its source, plus all benchmark graphs and the scripts that regenerate them from `bench/` data, are in the repository, and no figure lacks a recorded hardware and software configuration.

#### [TASK-20] Developer Documentation & API Freeze
- **Milestone**: Final (Dec 6)
- **Status**: Backlog
- **Priority**: `should` | **Area**: `docs`
- **Description**: Finalize the C/Zig API headers, setup guide, and operator runbook; tag `v1.0.0` on GitHub Releases.
- **Acceptance Criteria**: Done when someone else can build the project from scratch on a clean aarch64 or x86_64 Linux machine using only `README.md` instructions in under 10 minutes, and the public headers carry a stability note for v1.

---

## 4. Scope Tiers & Cut Lines

| Tier | Label | Meaning | Tasks |
|---|---|---|---|
| **Must** | `must` | The release fails without it. Together these form the **MVP: Two-Process Slice**: a real MoE runs split across two engine processes (RDMA on `gx10-03`, TCP between the desktop and `gx10-03`), loaded through `io_uring`, shown in a browser replay. | 01, 02, 03, 04, 05, 06, 07, 08, 10, 15, 16, 18, 19 |
| **Should** | `should` | The headline features beyond the MVP. Planned and expected, but droppable as a group. | 09, 11, 14, 17, 20 |
| **Could** | `could` | Stretch work, attempted only when ahead of schedule or when more hardware frees up. | 12, 13, 15b, 16b |

**If we fall behind, we cut first:** the **Stretch** cut (`cut-1`): NVMe KV offload, compressed expert streaming, the multi-Spark run, and live telemetry.

**Cut lines.** Cuts are applied in order. A cut task moves to `Backlog` with a `cut` note and is listed as future work in TASK-19.

| Cut | Tag | Name | Tasks dropped | Trigger |
|---|---|---|---|---|
| **First cut** | `cut-1` | **Stretch** | 12, 13, 15b, 16b | Alpha (`v0.2.0-alpha`) not tagged by Tue Nov 10. TASK-15b alone drops if neither `gx10-01` nor `gx10-02` is free by Sun Nov 22. |
| **Second cut** | `cut-2` | **Speculation & Swarm** | 09, 11, 14, 17 (engine runs with resident experts and a static manifest) | TASK-11 not in `Review` by Wed Nov 18 |
| **Emergency** | (none) | **TCP Floor** | Replace TASK-07 RDMA with the TASK-02 shared-memory and TCP backends for all later tasks; report those numbers as such | RDMA loopback on `gx10-03` not passing by Wed Nov 4 |

TASK-20 is `should` but untagged: it is never cut, only reduced to a minimal build guide.

**Out of scope for v1** (recorded so they are not silently added): 8-node and 1-trillion-parameter runs (only 3 Sparks exist), OpenAI-compatible REST server, approximate expert skipping, hardware DE tuning beyond TASK-13, GPU-initiated RDMA (IBGDA), training or fine-tuning, native Windows or macOS engine builds.

---

## 5. Workboard Snapshot

Columns: `Backlog` | `To Do` | `In Progress` | `Review` | `Done`

| Column | Tasks |
|---|---|
| **Backlog** | TASK-07 to TASK-20 (including 15b, 16b) |
| **To Do** | (none) |
| **In Progress** | (none) |
| **Review** | TASK-04, TASK-05, TASK-06 (all acceptance checks pass; Done once CI is green on `main`) |
| **Done** | TASK-01, TASK-02, TASK-03 |

| Area | Tasks | Count |
|---|---|---|
| `code` | 01, 02, 03, 07, 08, 10, 11, 12, 13, 14, 17 | 11 |
| `architecture` | 06, 09 | 2 |
| `benchmarking` | 05, 15, 15b | 3 |
| `webgl` | 04, 16, 16b | 3 |
| `docs` | 18, 19, 20 | 3 |

| Priority | Count | Cut tag |
|---|---|---|
| `must` | 13 | none |
| `should` | 5 | 4 tagged `cut-2` |
| `could` | 4 | all tagged `cut-1` |

**GitHub labels to create**: `must`, `should`, `could`, `code`, `architecture`, `benchmarking`, `webgl`, `docs`, `cut-1`, `cut-2`. Milestones: PoC, Alpha, Beta, RC, Final, with the dates above.

**Board rules** (solo + AI workflow)
- Work in progress is capped at 2 tasks.
- A task moves to `Review` when its PR is open with the acceptance criteria checked off. Review is an AI code review plus a self-review the next day; benchmark claims need the raw output linked.
- A task moves to `Done` only after merge and with evidence (test output, benchmark JSON, or screenshot) linked.
- Every Sunday milestone ends with a 15-minute check against the cut triggers in Section 4, and a check on whether `gx10-01` / `gx10-02` have freed up.

---

## 6. Dependencies & Critical Path

| Task | Depends on | Unblocks |
|---|---|---|
| TASK-01 | WSL2 setup, SSH key on `gx10-03` | 02, 03, 08, 10 |
| TASK-02 | 01, 06 (format) | 07, 10 |
| TASK-03 | 01 | 08, 12, 13 |
| TASK-04 | (none) | 16 |
| TASK-05 | SSH key on `gx10-03` | 07, 13, 15 (re-baseline targets) |
| TASK-06 | (none) | 02, 10 |
| TASK-07 | 02, 05 | 10, 11, 14, 15 |
| TASK-08 | 03 | 10, 13 |
| TASK-09 | 10 | 14, 17 |
| TASK-10 | 07, 08 | 11, 14, 15 |
| TASK-11 | 10 | 14 |
| TASK-12 | 03, 10 | (none) |
| TASK-13 | 05, 08 | (none) |
| TASK-14 | 07, 11 | 15 (if shipped) |
| TASK-15 | 10, time agreed with co-tenant | 16, 19 |
| TASK-15b | 15, a free and cabled second Spark | 19 |
| TASK-16 | 04, 15 | 18 |
| TASK-17 | 10 | 15 (hardening) |
| TASK-18 | 16 | Final |
| TASK-19 | 15 | Final |
| TASK-20 | all code tasks | Final |

**Critical path (MVP)**: TASK-05 → TASK-01 → TASK-02 → TASK-07 → TASK-10 → TASK-15 → TASK-16 → TASK-18.

---

## 7. Risks, Research-Verified Constraints & Fallbacks

### 7.1 Top 3 Risks (each tested in the PoC)

| # | Risk | PoC test (task) | Trigger date & condition | Fallback |
|---|---|---|---|---|
| **R1** | **Zig is new to the only engineer.** Zig is pre-1.0, with breaking changes between versions and fewer learning resources; combining it with C, CUDA, liburing, and verbs could stall the schedule. | TASK-01 toolchain gate: `build.zig` compiles C, links `liburing` and `libcudart`, calls CUDA on the 5070 Ti, cross-compiles to aarch64, and runs on `gx10-03`. | **Wed Oct 28**: gate not passing, or the TASK-01 matmul not working by Fri Oct 30. | Write the engine core in C11 with CMake (reusing the existing CLion project). No task changes, because every subsystem already sits behind a C ABI; Zig can return later for the scheduler only. |
| **R2** | **Only one node, and it is shared.** `gx10-01` and `gx10-02` are busy, so real node-to-node RDMA may never be available; `gx10-03`'s ConnectX-7 ports may be unlinked, which can block even RDMA loopback; the other student's jobs compete for memory, GPU, and disk. | TASK-05 probe: ConnectX-7 port state, loopback `ib_write_lat` on `gx10-03`, and realistic free memory; TASK-02 TCP backend between the desktop and `gx10-03`. | **Fri Oct 30**: loopback `ib_write_lat` does not run on `gx10-03`, or less than about 40 GB of memory is realistically free. **Memory condition already met on Oct 26 (about 11 GB free).** | RDMA code is developed against Soft-RoCE (`rdma_rxe`) and performance runs use the shared-memory and TCP backends if loopback fails (Emergency cut); for memory, cap each engine process at 4 GB on `gx10-03`, develop and test models on the desktop (64 GB RAM), use OLMoE-1B-7B-class models on the Spark, and ask the co-tenant for a window with vLLM paused for the soak; label every benchmark with the hardware and co-tenant load it ran under. |
| **R3** | **NVMe expert streaming is too slow or unsupported.** Direct I/O alignment may not be supported by `gx10-03`'s filesystem, or throughput may fall far below what streaming needs. **Retired Oct 27 for single-drive reads:** ext4 reports DIO alignment and the reader reaches 7.2 GB/s, matching `fio`; contention with GPU kernels is still untested. | TASK-03 reader versus the `fio` O_DIRECT baseline on `gx10-03`. | **Fri Oct 30**: O_DIRECT reads fail, or the reader reaches under 50% of the `fio` baseline. | Load every expert at startup and run resident-only (the RC model fits in memory); TASK-12 and TASK-13 drop with the first cut, and TASK-11 prefetching targets the transport only. |

### 7.2 Other Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R4 | One engineer for 22 tasks in 6 weeks | High | High | Cut lines in Section 4 with dated triggers; AI pair for boilerplate, tests, and review |
| R5 | Co-tenant load makes `gx10-03` benchmarks noisy | High | Medium | Record co-tenant GPU and memory use with each run; repeat runs and report the median; agree on a quiet window for the soak |
| R6 | Lookahead recall below 80% at H = 2 | Medium | Medium | Report H = 1; exact-mode miss handling keeps output correct |
| R7 | Kernel code written on the 5070 Ti (12.0) behaves differently on GB10 (12.1) | Medium | Medium | Build for both `sm_120` and `sm_121`; run correctness tests on both; never publish desktop numbers as GB10 numbers |
| R8 | GPLv3 vs. proprietary NVIDIA libraries (CUDA runtime, nvCOMP) in distributed binaries | Medium | Medium | Add a GPLv3 section 7 linking exception for NVIDIA libraries before the first release; check Lumabri's license before vendoring |
| R9 | Public demo exposes a live endpoint or lab machine details | Low | High | Pages ships recorded traces only; live `wss://` is operator-configured; no IP addresses or credentials in the repo |

### 7.3 Research-Verified Constraints

The blueprint PDF contains several figures that the verification PDF shows are not established for GB10. This plan follows the verification report:

| Topic | Blueprint claim | Verified boundary | Plan impact |
|---|---|---|---|
| CUDA target | SM120 / `sm_120` | GB10 is compute capability 12.1; target `sm_121` and query at runtime (the 5070 Ti is 12.0) | TASK-05 probe; build both |
| Tensor Memory | `tcgen05.mma` and TMEM ping-pong tiles | **Confirmed absent (Oct 27):** ptxas rejects `tcgen05.alloc` for `sm_121a` and `sm_120a` but accepts it for `sm_100a`. Shared memory is 99 KiB per block | Kernels use 12.x-style block-scaled MMA with shared-memory tiles |
| RDMA link | 400 Gbps ConnectX-7 | DGX Spark lists 200 Gbps | TASK-07 targets are relative to measured `ib_write_bw` |
| Hardware DE | 539 to 600 GB/s on GB10 | nvCOMP DE FAQ lists B200/B300/GB200/GB300, not GB10. **Confirmed absent (Oct 27):** `MEM_DECOMPRESS_ALGORITHM_MASK = 0` on `gx10-03` | TASK-13 uses software decompression only, and stays `could` |
| GPUDirect RDMA | NIC writes straight into GPU buffers | `GPU_DIRECT_RDMA_SUPPORTED = 0` on `gx10-03` (Oct 27); the GPU reads pageable host memory, which is the same LPDDR5X | TASK-07 registers host (`Arena`) memory with `ibv_reg_mr`; no `nvidia-peermem` |
| DE codecs | Bitcomp / GDeflate | Fixed-function DE supports LZ4, Snappy, Deflate; Bitcomp and GDeflate are SM-only | TASK-13 codec choice |
| TMEM target | NVMe → DE → TMEM | No documented TMEM destination; DE ends in CUDA memory buffers | TASK-13 pipeline |
| Immediate data | Auto-increments a counter | CQE carries the 32-bit value; CPU must publish readiness | TASK-07 design |
| GPU visibility | CPU atomics make data visible to GPU | Running kernels need a separate CUDA ordering contract | TASK-07, TASK-10 |
| NVMe rate | 14.0 GB/s single drive | Not established by primary sources; the 90 GiB/s figure is an 8-drive array. `gx10-03`'s drive is PCIe Gen 4 x4 (about 7 GB/s ceiling), not Gen 5 | TASK-03 uses a `fio` baseline |
| H = 2 / 3 recall | Above 94% with over-provisioning | Only H = 1 recall (85 to 97%) is published | TASK-11 measures recall |
| Expert skipping | Negligible quality loss | Not semantics-preserving | Out of scope for v1 |
| Packet size | 7,424 bytes, 128-byte aligned | Offsets omit 16 bytes of scales; true size 7,440 | TASK-06 fixes the layout |

---

## 8. Definition of Done (All Tasks)
- Code merged to `main` through a PR; CI green.
- Acceptance criteria evidence linked in the task (logs, JSON, screenshots).
- Published benchmarks taken on `gx10-03` (or another Spark), never WSL2; desktop results are labeled as desktop results. Each includes hardware, firmware, kernel, driver, CUDA, and nvCOMP versions, plus co-tenant load.
- Public-facing docs updated if behavior or APIs changed.
- No web deliverable introduces a downloadable executable or installer.
- No credentials, SSH keys, or tailnet IP addresses committed to the repo (machine nicknames like `gx10-03` are fine).

---

## 9. Ramp-Up Research (Week 0)

Budget about one day on Zig and half a day on each of the others, spread across the PoC week as each task needs it.

**Zig (for TASK-01, Risk R1)**
- [Zig Language Reference](https://ziglang.org/documentation/master/) for the pinned version (switch the URL to the pinned version number), and [ziglang.org/learn](https://ziglang.org/learn/).
- [Ziglings](https://codeberg.org/ziglings/exercises): small exercises; do roughly the first 40.
- Topics to learn first, in order: error unions and `try`, slices and pointers, explicit allocators (`std.mem.Allocator`, arenas), `defer` / `errdefer`, `comptime`, `extern struct` and C interop (`@cImport`, `translate-c`), `build.zig` (`addExecutable`, `addCSourceFiles`, `linkSystemLibrary`, `linkLibC`), and cross-compiling (`-Dtarget=aarch64-linux-gnu`).
- Zig is pre-1.0: code and tutorials written for older versions often fail to compile. Check the version a tutorial targets.

**Linux I/O (for TASK-03)**
- [Lord of the io_uring](https://unixism.net/loti/) tutorial, especially fixed buffers and SQPOLL.
- `man io_uring_setup`, `man io_uring_prep_read_fixed`, and `O_DIRECT` alignment in `man 2 open`.
- `fio` basics for building a baseline.

**RDMA (for TASK-05, TASK-07)**
- rdma-core's [`rc_pingpong.c`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/examples/rc_pingpong.c): the canonical RC QP example.
- NVIDIA's RDMA Aware Networks Programming guide (QP state transitions INIT → RTR → RTS, memory registration).
- `perftest` tools (`ib_write_lat`, `ib_write_bw`) and Soft-RoCE (`rdma_rxe`) for testing without a live link.

**MoE and formats (for TASK-06, TASK-08, TASK-10)**
- Top-k gating and expert FFNs: the Switch Transformer and Mixtral papers, and the OLMoE paper for a small open reference model.
- NVIDIA's "Introducing NVFP4" blog post, and the GGUF spec in the llama.cpp repository.

**Blackwell platform (for TASK-05, R7)**
- [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) and the CUDA compute capability table (GB10 = 12.1, RTX 5070 Ti = 12.0).
- CUTLASS's Blackwell documentation, noting which features are SM100-only versus SM120-family.
- The `Research/` PDFs in this repo, verification report first.