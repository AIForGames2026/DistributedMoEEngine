# Project Plan & Roadmap: TensorFold-Swarm

> Plan baseline: Monday, Oct 26, 2026. Milestone dates fall on Sundays; each milestone closes at end of day on its date.
> Repository: https://github.com/AIForGames2026/DistributedMoEEngine
> Team: one engineer (Luke Hinojosa) working with an AI pair programmer (Claude). All tasks are owned by Luke.
> Board: https://github.com/orgs/AIForGames2026/projects/3 (issues #1 to #35 mirror the tasks below; #23 is TASK-21, added after the PoC, #24 to #31 are TASK-22 to TASK-29, #32 to #34 are TASK-30 to TASK-32, opened Nov 15, #35 is TASK-33, opened Nov 16, and #36 to #41 are TASK-34 to TASK-39 #42 is TASK-40 and #43 is TASK-41, opened Nov 17). Live demo: https://aiforgames2026.github.io/DistributedMoEEngine/
> Sources: `Research/GB10 Distributed MoE Engine.pdf` (system blueprint) and `Research/NVIDIA Blackwell Hardware Decompression Engine.pdf` (platform verification). Where the two disagree, the verification report wins. See Section 7.

### Hardware

| Machine | What it is | Availability | Role in this project |
|---|---|---|---|
| `gx10-03` | NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, ConnectX-7) | **Now**, shared with another student (about 11 GB of memory free while their vLLM job runs) | Primary engine node and all published benchmarks |
| `gx10-02` | NVIDIA DGX Spark (GB10) | **Free since Nov 11** (the GLM 5.3 job ended); set up with `~/lukehinojosa/` | Multi-node RDMA (TASK-15b) |
| `gx10-01` | NVIDIA DGX Spark (GB10) | **Busy** (GLM 5.3 until Nov 10, another model server since Nov 11) | A third node if it frees up (TASK-15b) |
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
| **PoC** | Nov 1, 2026 | Zig engine loop, mock token dispatch, io_uring prototype, platform probe, wire format, & WebGL mockup | `v0.1.0-poc` (released Oct 27) |
| **Alpha** | Nov 8, 2026 | RDMA transport (single-node loopback first), quantized expert loader, & first two-process forward pass | `v0.2.0-alpha` (released Nov 10) |
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
  - [x] Set up key-based SSH to `gx10-03` (`docs/BUILDING.md`, "Lab machine access"); done Oct 26 for all three Sparks.
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
- **Status**: Done
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
- **Status**: Done
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
- **Status**: Done
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
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement the RDMA backend of the TASK-02 `Transport` with libibverbs Reliable Connection QPs over ConnectX-7, using one-sided `IBV_WR_RDMA_WRITE_WITH_IMM` into pre-registered remote slot rings. Bootstrap QPN, PSN, GID index, remote address, and rkey from a launcher manifest. The CPU publishes receiver-side readiness after it sees the CQE; the immediate value does not increment memory by itself. With one free node, this is developed and accepted as two processes on `gx10-03` talking through its ConnectX-7 (HCA loopback). The Sparks are already direct-cabled in a full mesh at 200 Gb/s, so real two-node runs need only `gx10-01` or `gx10-02` to be free (TASK-15b).
- **Acceptance Criteria**: Done when two processes on `gx10-03` pass the TASK-02 test suite unchanged on the RDMA backend, single-packet one-way latency p50 is within 1.5x of loopback `ib_write_lat` for the same size, sustained bandwidth reaches at least 90% of loopback `ib_write_bw`, and no payload slot is reused before its send completion. If TASK-05 shows RDMA loopback cannot run on `gx10-03`, the same suite passes against Soft-RoCE (`rdma_rxe`) and the Emergency cut in Section 4 applies for performance numbers.
  - **From the PoC (Nov 2)**: keep the ordered goodbye (rank 0 first; rank 1 tears down after rank 0), since the relayed TCP path showed that close-time ordering cannot be assumed; and keep CRC32C on the hardware instructions (or offload it), because software CRC dominated per-token cost on ARM.
  - **Results (Nov 5)**: all acceptance checks pass on `gx10-03` over HCA loopback (`roceP2p1s0f0`). `csrc/tf_rdma.c` (verbs) under `src/transport/rdma.zig` (`Backend`); built with `zig build -Drdma=true`; run with `dispatch rdma` (or `rdma-expert` / `rdma-router`). The TASK-02 dispatch check and the PoC `--moe` check pass 20 of 20 runs each with both processes exiting 0. One-way latency p50 4.34 to 4.39 us against a 4.68 us target (`ib_write_lat` 3.12 us today); one-direction bandwidth 100.7 to 102.0 Gb/s against 92.4 Gb/s (`ib_write_bw` 102.65 Gb/s today). Bandwidth needs 256 slots per ring (64 slots reach only 74 to 83 Gb/s, limited by credits in flight), so the `rdma` modes default to 256. A send buffer is reused only after its send completes. Queue-pair details are swapped over TCP instead of a launcher manifest (the manifest arrives with TASK-09). Not done yet: striping across both devices of a port and node-to-node runs (TASK-15b). See `bench/transport/2026-11-05.md`.

#### [TASK-08] Quantized Expert Loader (GGUF + NVFP4)
- **Milestone**: Alpha (Nov 8)
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement a model loader that parses GGUF (Q8_0 and MXFP4 tensor types) and NVFP4 checkpoints (E2M1 values, E4M3 scale per 16 values, FP32 tensor scale) and loads only the experts this engine process owns, reading through the TASK-03 `io_uring` path.
- **Acceptance Criteria**: Done when the engine parses a multi-expert model chunk, loads only this process's experts into memory, validates tensor shapes and checksums against the file metadata, and dequantized values match the reference harness within format tolerance.
  - **Results (Nov 5)**: all acceptance checks pass. `src/gguf.zig` and `src/safetensors.zig` parse the headers; `src/quant.zig` decodes F32, F16, BF16, Q8_0, MXFP4, NVFP4 (GGUF blocks), and ModelOpt NVFP4 (E4M3 per 16 values plus the f32 `weight_scale_2`); `src/loader.zig` checks every expert tensor's shape against the model metadata (GGUF keys or `config.json`), plans aligned reads of only the owned experts, and checks each against a CRC32C index; `tfload` (`zig build tfload`) runs it through the io_uring reader. On `gx10-03`, half of OLMoE-1B-7B-0924-Instruct Q8_0 (experts 0 to 31, 3.42 GB of 7.36 GB) loads at 7.0 GB/s (97% of `fio`) with every expert's CRC32C checked in 0.13 s; 18 sampled expert weights are bit-identical to llama.cpp's gguf-py dequantizer. MXFP4, GGUF NVFP4, and ModelOpt NVFP4 are bit-identical on fixtures from `tools/ref/ref_quant.py` (unit tests and CI). Deviations: neither format stores checksums, so the index is a sidecar (`MODEL.tfsum`, written once by `tfload index`; the file is checked once against Hugging Face's SHA-256 at download); safetensors support is single-file (sharded checkpoints not yet); no public OLMoE checkpoint uses MXFP4 or NVFP4, so those formats are checked on fixtures only. See `bench/storage/gx10-03-2026-11-05.md`.

#### [TASK-10] Top-k Expert Gating & First Two-Process Forward Pass
- **Milestone**: Alpha (Nov 8)
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Implement the gating router (softmax over top-k logits, starting with k = 1 and generalizing to k = 8) that sends each token to local or remote experts, runs the expert FFN, and combines weighted results. Combine writes each expert contribution to its own slot and reduces locally; overlapping RDMA writes do not sum. Expert placement comes from a static manifest file until TASK-09 lands.
- **Acceptance Criteria**: Done when a token sequence completes a forward pass split across 2 engine processes (on `gx10-03` over RDMA, and between the desktop and `gx10-03` over TCP) for the TASK-06 synthetic MoE and one small public MoE (for example OLMoE-1B-7B), with output logits matching the PyTorch reference within a documented tolerance (target: max abs error at or below 1e-2 in FP8 paths) and identical top-1 next tokens on the golden prompts.
  - **Progress (Nov 2, PoC core check)**: the TASK-06 synthetic MoE runs split across 2 processes on CPU (`dispatch --moe`: rank 0 routes, rank 1 owns every expert) over shared memory, TCP loopback, and TCP desktop to `gx10-03`. Routing identical, logits max abs error 7.15e-7, top-1 16/16 on every path. Left: GPU kernels, the RDMA path (needs TASK-07), and OLMoE-1B-7B (needs TASK-08). See `POC.md` and `bench/transport/2026-11-02.md`.
  - **Progress (Nov 5)**: the same synthetic MoE now also passes over RDMA on `gx10-03` (`dispatch rdma --moe`, 20 of 20 runs, logits max abs error 7.15e-7, top-1 16/16). Left: GPU kernels and OLMoE-1B-7B (needs TASK-08).
  - **Progress (Nov 5)**: TASK-08 loads OLMoE-1B-7B's experts (Q8_0, owned experts only). Left: GPU kernels, and running the OLMoE expert FFNs from the loaded weights (attention, norms, and router weights still need loading too).
  - **Results (Nov 5)**: all acceptance checks pass. `src/olmoe.zig` runs OLMoE-1B-7B (attention with QK norm, rotary embeddings, softmax-then-top-8 routing, SwiGLU experts) from the Q8_0 GGUF; `olmoe` (`zig build olmoe`) splits it across two processes, rank 0 holding everything but experts 32 to 63, which rank 1 holds, from a static placement manifest (`--placement`, default: halves). Against transformers' `OlmoeForCausalLM` on the same weights (`tools/ref/ref_olmoe.py`), three golden prompts give logits within 3.77e-5 (target 1e-2), every top-1 token identical (43/43), and every routing choice identical, over RDMA on `gx10-03` and over TCP from the desktop to `gx10-03`, as well as over shared memory and in one process; the synthetic MoE passed on all paths earlier. With `--gpu` (`-Dgpu=true`), every Q8_0 matmul runs on CUDA (`csrc/cuda/kernels.cu`): the 17-token prompt takes 0.41 s on the RTX 5070 Ti (CPU 4.5 s) and 0.67 s split over RDMA on GB10 with the weights used in place (CPU 2.9 s), with the same accuracy. A tiny random OLMoE with transformers' outputs (`ref_olmoe.py --tiny`) checks the whole forward pass in the unit tests and CI. Notes: the 1e-2 target is met with float32 payloads, sent as two packets per expert call (`flags` now numbers payload encodings and chunks; `docs/wire-format.md`); FP8 (the wire format's default) and BF16 keep every top-1 token but put the logits off by up to 2.1 and 0.35. On the GPU, norms, attention scores, and routing still run on the CPU. One unexplained GPU failure (the first run on `gx10-03`, likely NaNs) in 16 GPU runs. Prompts are processed whole (prefill); there is no KV cache or token-by-token generation yet (needed for TASK-15). See `bench/olmoe/2026-11-05.md`.

#### [TASK-21] Direct Desktop-to-Spark Network Path
- **Milestone**: Alpha (Nov 8)
- **Status**: Done
- **Priority**: `should` | **Area**: `benchmarking`
- **Description**: Added after the PoC. Every desktop-to-`gx10-03` run so far went through Tailscale's DERP relay in NYC (33 ms round trip, about 5 ms per token pipelined), so cross-machine numbers measure the relay, not the transport. Find a direct path: Tailscale direct UDP (41641) through the campus network, a subnet route, or another link the lab allows.
- **Acceptance Criteria**: Done when `tailscale ping` (or the chosen route) reports a direct connection and `dispatch router --packets 4096` desktop to `gx10-03` is re-measured and recorded in `bench/transport/`; if no direct path is possible, the reason is recorded and TASK-10 and TASK-15 label their cross-machine numbers as relay-bound.
- **Results (Nov 5)**: direct path found, plain TCP only. `gx10-03` dials the desktop's campus address, so the Spark runs the router and the desktop the expert. `dispatch router --packets 4096` takes a median 1.02 s (248 µs per token, 0.49 Gb/s) against 49.3 s over the relay in the same setup; one-way latency p50 is 772 µs against 18.5 ms. OLMoE-1B-7B split over this path, with the expert in WSL2 behind a small forwarder, keeps the same accuracy and takes 3.95 s for the 17-token prompt, against 24.3 s over the relay. Tailscale still cannot go direct: every UDP flow between the machines is cut after 3 packets per side at any rate, apparently by a campus firewall, so `tailscale ping` stays on DERP. Opening UDP would be a request for campus IT, and no one in charge of the network has been asked about the TCP path yet. TASK-15's desktop-to-Spark runs can use this path, and should keep the expert's port open only during a run, since the TCP transport has no authentication. See `bench/transport/2026-11-05-direct-path.md`.

#### [TASK-23] Alpha Vertical Slice: Guided Mission & Test Plan
- **Milestone**: Alpha (Nov 8)
- **Status**: Done
- **Priority**: `must` | **Area**: `webgl`
- **Description**: Added Nov 5 for Testing Session 1 (week of Nov 9). Classmates cannot run the engine, and the slice is simulated by design (the instructor knows the scope is deep and does not need it changed), so the page has to teach the idea with no developer help. On first load the inspector runs a guided mission: a short intro, three experiments (watch it work, remove prediction, split across the relay) with a check question, a final challenge (get two machines to 200 tok/s, which only the direct link can do), and a report card with a result line, Play again, and Exit. The numbers come from the existing simulator, and `npm test` fails if a lesson stops holding. A panel separates recorded engine results from the simulation. Top-k and recall stay locked during the mission.
- **Acceptance Criteria**: Done when a first-time tester finishes the mission unaided on the live GitHub Pages site, the site has been loaded on a second machine, `TestPlan.md`, the release notes, and the redeploy steps are in the repo, and `v0.2.0-alpha` is tagged.
- **Results (Nov 13)**: acceptance met. Testing Session 1 ran asynchronously on the class forums (Nov 13 to 15). In the first six replies, all six testers finished unaided, with a median of 3.1 minutes, and the 3D view read clearly (red cubes mentioned by 3 of the 4 who answered). But most could not say what they had seen: 1 of 4 explained why data travels between machines, 1 of 4 reasoned their way to the final fix, and the locked controls made the challenge a process of elimination. See `bench/playtest/session-1.md`. Superseded as the main page by TASK-30's chat harness and live diagram; the mission stays as an optional tab, or goes, when TASK-18 ships.

---

### Milestone 3: Beta (Target: Nov 22)

#### [TASK-09] Swarm Node Tracker (after Lumabri / Colibri)
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `should` | **Area**: `architecture` | **Tag**: `cut-2`
- **Description**: Replace the static placement manifest with a tracker as the control plane: live process and node registry, health heartbeats, and the expert ownership map the dispatcher reads. Token payloads never pass through it. Changed Nov 11: Lumabri turned out to be Apache-2.0 (compatible) but a large, fast-moving C project whose tracker beats every 10 s on port 7300, too slow for the 5 s target below, so the tracker is written in Zig (`src/tracker.zig`, TCP 47300) after Lumabri's index-only design instead of vendored.
- **Acceptance Criteria**: Done when a new engine process joins and its expert assignment appears in every peer's ownership map within 2 seconds, a killed process is marked unhealthy within 5 seconds, and the dispatcher refuses to route to an unhealthy process.
- **Results (Nov 11)**: acceptance met across two Sparks: tracker on `gx10-02`, router on `gx10-02`, expert on `gx10-03` over RDMA. Processes join over TCP with their address and experts, send a heartbeat every 250 ms, and get the whole map on every change. A process is unhealthy when its connection closes without a goodbye or after 1 s without a heartbeat. `olmoe ... --tracker IP` replaces `--host` and `--placement`: the router waits for a healthy expert, uses the experts it announced, and stops with `PeerUnhealthy` instead of routing to it once it is unhealthy. Measured: a join reached every peer in 2 ms (bench with 16 peers on the other node: p50 3.4 ms, max 10.9 ms); a SIGKILLed expert was marked unhealthy in 52 ms and a SIGSTOPped one in 0.79 s, and the router refused 10 ms after each map arrived; a router started while the only expert was stopped waited 4.8 s and connected when it resumed. With the tracker, generation stays at 55 tokens/s and passes every check. Limits: the tracker is a single point of failure (clients then fail closed), there is no authentication, and the engine still runs two ranks. See `bench/tracker/2026-11-11.md`.

#### [TASK-11] Speculative Lookahead Router
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `should` | **Area**: `code` | **Tag**: `cut-2`
- **Description**: Build a lightweight low-rank or cascaded predictor (following SpecPrefetch / Director) that predicts expert sets ahead (H = 1 first, H = 2 target) and issues RDMA pre-arming and NVMe prefetch commands through the Prefetch Command Ring. The native router stays authoritative: predictions only drive transfers, never which experts run. Prefetch keys are `(layer, expert)` pairs. Predictor training can run on the desktop's 5070 Ti.
- **Acceptance Criteria**: Done when prefetches are issued before the target layer starts in at least 95% of steps, measured expert recall at H = 2 with k + 2 candidates is at least 80% on two evaluation datasets (or H = 1 results are reported if H = 2 misses), predictor compute overhead is under 2% of layer time, and outputs are bit-identical with the predictor on versus off.
- **Results (Nov 10)**: acceptance met on the desktop and over RDMA on `gx10-03` (recall 83.6% in generation, guessing 0.56% of layer time, bit-identical, every hint before its layer). The predictor needs no training: after layer S's attention it applies layer S + 2's own norm and router to S's hidden state and keeps the top 8 + 2 experts (layers 0 and 1 are guessed from the token embeddings). Recall at H=2 is 82.7% on WikiText-2 and 82.2% on HumanEval (16,384 tokens each, `tools/ref/lookahead_study.py`), and 83.6% during generation in the engine. Guessing takes 1.62% of layer time in one process and 1.33% in two, outputs are bit-identical with it on versus off (prompts and 3 x 64 generated tokens), and rank 1 received every layer's prefetch hint (`flags` bit 6) before that layer's first expert call, 6.1 ms early on average. With `--cache N`, a process holds its experts in N slots that a worker thread fills from the model file through the io_uring reader, on hints and on demand (layer 0 stays loaded, since its guesses are weak, 19 to 26%): with a quarter of the experts in memory, hints cut misses by about two thirds (45.7% to 14.8% on `gx10-03`), but throughput rises only 12% (8.0 to 9.0 tokens/s, against 20.4 with every expert loaded) because the worker reads one expert at a time and both ranks share one drive; TASK-14's parallel and prioritized loads are the next step. Output is unchanged in every run. See `bench/olmoe/2026-11-10-lookahead.md`.

#### [TASK-14] Misprediction & Miss Handling (Exact Mode)
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `should` | **Area**: `code` | **Tag**: `cut-2`
- **Description**: Handle experts the lookahead missed, without changing model output: cancel obsolete speculative requests, promote true misses to a high-priority queue, run resident experts while missing ones arrive, and fetch from a peer process's memory over RDMA READ when that beats local NVMe. An approximate skip-low-weight mode is out of scope (Section 4).
- **Acceptance Criteria**: Done when, under forced 10% mispredictions for 30 minutes, the engine produces tokens continuously without crash, deadlock, or memory growth, and output is bit-identical to a no-speculation run.
- **Results (Nov 11)**: acceptance met over RDMA on `gx10-03`: 30.1 minutes with 10% of guesses forced wrong, 128 rounds at 13.6 tokens/s, tokens and logits identical to a predictor-off round in every round, resident memory +0.00%, exit 0. Hints carry the position they are for and loaders drop the ones whose layer has started (31,025 in the soak); misses go on a queue the loaders empty before hints; `--loaders N` (default 2) reads several experts at once; both ranks run the calls whose experts are ready and the rest as they arrive, adding results in a fixed order so output does not change. Every cache setup in the sweep beat TASK-11 by 9 to 28% (128 slots: 9.0 to 10.2 tokens/s; 256: 13.1 to 14.6), but the shared drive is still the limit, so more loaders or hints stop helping past two. Fetching from a peer over RDMA READ is left out: the ranks own disjoint experts, so a peer never holds a missing one; it needs replicated experts (TASK-15/16). See `bench/olmoe/2026-11-10-miss-handling.md`.

#### [TASK-12] Tiered NVMe KV Cache Offloader
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `could` | **Area**: `code` | **Tag**: `cut-1`
- **Description**: Implement a ring-buffer KV manager that pins attention sink tokens (first 128) and the most recent 8,192 tokens in memory, and offloads cold middle context to NVMe via `io_uring`, paging it back on demand with prefetch. Changed Nov 11: OLMoE's context is 4,096 tokens, so this task runs on Qwen3-30B-A3B (Q8_0, about 32 GB; 32k native context, 128k with YaRN factor 4), which needs Qwen3-MoE support in the engine (grouped KV heads, per-head QK norm, renormalized top-8 routing, YaRN) and attention on the GPU, since a 128k prefill on the CPU takes hours. The acceptance criteria were rewritten the same day: the old target, under 15% decode slowdown against an all-RAM cache, cannot be met with exact full attention. Every new token reads every cached position, so at 128k the offloaded middle (about 23.6 GB in f32) takes about 3.4 s per token to read at the drive's 7 GB/s, against about 0.1 s from RAM. Compact formats (TASK-27) and saved prefixes (TASK-28) are follow-ups that make this faster and easier to test, not fixes.
- **Acceptance Criteria**: Done when (1) Qwen3-30B-A3B with YaRN passes TASK-10's checks (logits within 1e-2 of transformers, top-1 identical, every routing choice identical) and TASK-24's greedy generation check on the golden prompts; (2) a 128k-token context runs with the in-memory KV cache capped to the first 128 plus the most recent 8,192 positions, and its tokens and logits match an all-RAM run at the same length exactly (both fit on `gx10-02`); (3) during decode the offloaded part is read at 5 GB/s or more, with reads overlapping compute; and (4) decode speed and time to first token at 128k, offloaded and all-RAM, are recorded in `bench/`.
- **Results (Nov 11)**: acceptance met on `gx10-02` with Qwen3-30B-A3B (Q8_0, `--yarn 4`). (1) Against transformers 4.57.1 with YaRN: logits within 2.65e-4, top-1 37/37, every routing choice identical, and all 3 x 32 greedy tokens match `generate()`. (2) A 131,040-token prompt (the repository's own text) plus 32 generated tokens, with only the first 128 and the most recent 8,192 positions in memory (3.07 GB with staging buffers, against 25.77 GB for the whole cache), produced the same 32 tokens and logits as the all-RAM run bit for bit. (3) Decode read the offloaded middle at 6.31 GB/s (24.1 GB per token). (4) Time to first token 1,918 s in RAM and 1,947 s offloaded; decode 0.230 s per token in RAM and 3.83 s offloaded, bound by the drive as estimated. Engine changes: Qwen3-MoE support (grouped KV heads, per-head QK norm, renormalized top-k, YaRN), CUDA attention that reads the cache in place through a tile table (`tf_attention_wide`, 4.6 to 5.9x faster than the first kernel), the tiered cache (`src/kv_tier.zig`, O_DIRECT writes, io_uring reads one layer ahead), `olmoe --context N` for long prompts, expert calls batched by expert with one GPU call per layer, and a threaded router. With the last two, a 16k prefill on GB10 takes 95.7 s, down from 139.1 s; experts still take half of it, so a tiled Q8_0 matmul for many tokens is the next prefill speedup. See `bench/olmoe/2026-11-11-long-context.md`. Follow-up the same day:
  - Speedups: a tiled matmul, a decode matmul, experts gathered and combined on the GPU, the router on the GPU, threaded per-token loops, tensor-core attention for prefill (f16 inputs split in two, so results stay close to f32), and a decode attention kernel.
  - Time to first token: 21.0 s at 16k (95.7 s before), 540 s at 128k in RAM (1,918 s), and 574 s offloaded (1,947 s).
  - Decode: 0.047 s per token at 16k (0.079 s) and 0.191 s at 128k in RAM (0.230 s).
  - Offloaded and all-RAM runs still match bit for bit, and the goldens pass (logits within 1.87e-4).
  - At 16k the tensor cores move generated tokens' logits by up to 0.85 against the f32 kernel, with identical tokens; a different order of f32 sums alone moves them by 0.20.
  - See `bench/olmoe/2026-11-11-speedups.md`.
  - Second batch: Q8_0 matmuls on tensor cores (8-bit weights exact in f16, inputs scaled per row by a power of 2 and split in two), and one wait per small GPU call. Time to first token 17.5 s at 16k, 510 s at 128k in RAM, and 547 s offloaded; decode 0.046 s per token at 16k and 0.181 s at 128k. Offloaded runs still match bit for bit. At 16k the generated tokens differ from the tiled f32 matmul's, as they also do between two f32 kernels that only add in a different order. See `bench/olmoe/2026-11-11-speedups-2.md`.

#### [TASK-13] Compressed Expert Streaming (nvCOMP, Hardware DE Only If Present)
- **Milestone**: Beta (Nov 22)
- **Status**: Backlog (`cut`, Nov 11)
- **Priority**: `could` | **Area**: `code` | **Tag**: `cut-1`
- **Description**: Store experts losslessly compressed on NVMe and decompress them into CUDA-accessible memory before the FP4 kernel consumes them. Use the hardware Decompression Engine only if TASK-05 finds it on GB10 (NVIDIA documents it for B200/B300/GB200/GB300, not GB10), selecting `NVCOMP_DECOMPRESS_BACKEND_HARDWARE` so incompatible buffers fail instead of silently falling back. Otherwise use SM-based nvCOMP. The pipeline ends in device memory buffers; there is no documented NVMe to DE to TMEM path.
- **Acceptance Criteria**: Done when the four-way benchmark (NVMe only, GEMM only, NVMe + GEMM, NVMe + decompress + GEMM) is recorded on real NVFP4 expert weights with compression ratio, compressed-input and output GB/s, GEMM slowdown, and end-to-end expert-ready latency, and compression is enabled by default only if it improves expert-ready latency by at least 10% over raw reads.
- **Cut (Nov 11)**: measured first, and it cannot pass its own 10% bar. On OLMoE's real Q8_0 experts (53.5 MB sampled from all 48 expert tensors), the 8-bit values carry 7.68 of 8 bits, so the best any per-byte coder (nvCOMP's ANS included) can reach is 1.066x; lzma gets 1.056x. NVFP4 made from the same weights reaches at best 1.090x (lzma 1.072x). Even with free, fully overlapped decompression that saves at most 6.2% (Q8_0) or 8.3% (NVFP4) of expert read time, and only when experts stream from the drive (`--cache N`); GB10 has no hardware Decompression Engine. Listed as future work in TASK-19. See `bench/storage/2026-11-11-compression.md`.

#### [TASK-22] CPU-Only Expert Nodes
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `could` | **Area**: `code` | **Tag**: `cut-1`
- **Description**: Added Nov 3. Let a machine with no NVIDIA GPU (any x86_64 or ARM Linux or Windows box, starting with the desktop's i7-12700K) join as an expert node that runs its experts on the CPU. The PoC expert already does this for the synthetic MoE (`moe_ref.expert` behind `dispatch --moe`, over TCP, on Linux and native Windows); this task extends it to real weights: a per-process `cpu` backend flag in the placement manifest, CPU dequantization of the TASK-08 formats (Q8_0, MXFP4, NVFP4) to FP32 at load time, and a tiled CPU expert FFN. The wire format and transport do not change. This is the only supported way for non-NVIDIA hardware to take part (Section 4).
- **Acceptance Criteria**: Done when the TASK-10 small public MoE runs split between a GPU process on `gx10-03` and a CPU-only expert process on the desktop over TCP, with logits within TASK-10's tolerance of the PyTorch reference and identical top-1 tokens, and the CPU process's per-expert latency and tokens/second are recorded in `bench/` (labeled as desktop CPU results).
- **Results (Nov 11)**: acceptance met. Rank 0 ran on the GB10 GPU in `gx10-03`; rank 1 (experts 32 to 63) ran on the desktop's i7-12700K in WSL2, built without `-Dgpu`, over the direct TCP path. Logits were within 3.72e-5, top-1 43/43, every expert choice identical, and all 3 x 64 generated tokens matched `generate()`, in two runs. Desktop CPU results: 1.79 to 1.83 ms per expert call and 4.6 to 5.1 tokens/s. The RTX 5070 Ti on the same link takes 0.23 ms per call and gets 11.8 to 12.4 tokens/s, so the single-threaded CPU FFN is now the limit. The first runs got only 1.2 to 1.4 tokens/s, with a CPU or GPU expert alike, because they reached WSL2 through its `localhost` relay; relaying to the WSL2 VM's own address fixed it. The plan's FP32-at-load dequantization and manifest flag were not needed: any process without `--gpu` runs the same Q8_0 code on the CPU, and FP32 would take 4x the memory. See `bench/olmoe/2026-11-11-cpu-expert.md`.

#### [TASK-24] Token Generation Loop & In-Memory KV Cache
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `must` | **Area**: `code`
- **Description**: Added Nov 5. TASK-10 processes a whole prompt in one forward pass, and TASK-15's soak has to generate tokens for an hour, but no task owned the step between them. Add a per-layer KV cache in memory, one-token-at-a-time forward passes, greedy and seeded sampling, and an `olmoe generate` mode over every transport. TASK-12 (NVMe offload) stays separate and optional.
- **Acceptance Criteria**: Done when greedy generation over RDMA on `gx10-03` produces the same tokens as transformers' `generate()` for the three TASK-10 prompts (at least 64 new tokens each, F32 payload), per-token logits stay within TASK-10's tolerance, and memory stays flat across a 10-minute run. Results recorded in `bench/`.
- **Results (Nov 10)**: acceptance met. KV cache, one-token decode, greedy and seeded sampling, and `olmoe --generate N` / `--minutes M` on every transport. Over RDMA on `gx10-03` (two processes, F32 payload), all 3 x 64 greedy tokens match transformers' `generate()` with logits within 4.53e-5, and a 10.1-minute run generated 12,288 tokens at 20.4 tokens/s with identical tokens in all 64 rounds and flat memory (rank 0 +0.00%, rank 1 3,539 to 3,540 MB). The desktop 5070 Ti gives the same tokens and errors in one process and two. See `bench/olmoe/2026-11-10-generate.md`.

#### [TASK-26] Ampere GPU Expert Nodes (RTX 30-Series)
- **Milestone**: Beta (Nov 22)
- **Status**: Done
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 10. Most classroom desktops have RTX 3070-class GPUs (Ampere, compute capability 8.6, 8 GB). The Q8_0 matmul kernel is plain CUDA, so the build adds an `sm_86` target (`build.zig`) next to the Blackwell ones; nothing else assumes Blackwell. Ampere has no FP4 or FP8 tensor cores, so the planned NVFP4 kernels stay Blackwell-only and Ampere nodes keep the Q8_0 path. An 8 GB card holds one half of the OLMoE split (about 3.9 GB), not the whole model. Tested on the owner's laptop, whose RTX 3060 Laptop GPU is the same architecture with 6 GB.
- **Acceptance Criteria**: Done when OLMoE-1B-7B split between an Ampere GPU and a second machine passes the TASK-10 checks (logits within 1e-2, top-1 identical, every routing choice identical) with `TF_GPU_CHECK=1` finding no GPU result off by more than 1e-3, and the run is recorded in `bench/` labeled with the GPU.
- **Results (Nov 10)**: acceptance met. OLMoE-1B-7B split between the laptop's RTX 3060 Laptop GPU (rank 0, 3.94 GB on the GPU) and the desktop's 5070 Ti, over direct Tailscale TCP: logits within 3.53e-5, top-1 43/43, every expert choice identical, and `TF_GPU_CHECK=1` found no GPU result more than 1e-3 off the CPU. Without the check it takes 0.61 / 2.34 / 2.10 s per prompt (5 / 21 / 17 tokens), largely bound by the link (`dispatch` laptop to desktop: 1.69 ms one way, 0.18 Gb/s); the inspector gained a matching "Desktop + laptop (RTX 3060)" cluster. The 5070 Ti still passes with the `sm_86` target added. See `bench/olmoe/2026-11-10-ampere.md`.

---

### Milestone 4: Release Candidate (Target: Nov 29)

#### [TASK-15] Soak Test on `gx10-03` + Cross-Machine Run
- **Milestone**: RC (Nov 29)
- **Status**: Done (run Nov 16)
- **Priority**: `must` | **Area**: `benchmarking`
- **Description**: Two runs with a real public MoE whose weights fit in the memory left free on `gx10-03`. With about 11 GB free today, that means a small model such as OLMoE-1B-7B at 4 bits (about 4 GB); a Qwen3-30B-A3B-class model (about 17 GB) only if the co-tenant's vLLM job is paused for the run. **Soak**: two engine processes on `gx10-03`, each owning half the experts, linked by the RDMA backend, generating for 1 hour. **Cross-machine**: the same model split between the desktop and `gx10-03` over TCP, proving the engine runs on two physically separate, mixed machines. Agree on a time with the other student before the soak.
- **Acceptance Criteria**: Done when the soak sustains at least 10 tokens/second aggregate for 1 hour with zero crashes and flat resident memory (under 1% drift after warm-up), the cross-machine run generates correct tokens for 10 minutes (its throughput is recorded but not targeted, since it rides on Tailscale), and both runs' logs (configs, versions, counters, telemetry traces, co-tenant load) are archived under `bench/rc/`.
- **Results (Nov 16)**: acceptance met, on `gx10-02` instead of `gx10-03` (otherwise idle, so no co-tenant time was needed). OLMoE-1B-7B Q8_0, every expert loaded, F32 payload, GPU on both ranks, recorded with `--record`. **Soak:** `olmoe rdma` (two processes on `gx10-02` over RDMA loopback) ran 60.0 minutes at 30.8 tokens/s (577 rounds, each 30.0 to 30.9), both processes exited 0, every round's tokens and logits identical to the first after the prompt and `generate()` checks against transformers passed, resident memory +0.00% after round 1 on both ranks (sampled each minute: unchanged to the kilobyte), and both ranks' weights unchanged at exit. Its recording (110,979 steps, none dropped) accounts for every expert call (8,761,889 on rank 0, 8,407,903 on rank 1, equal to the engine's counters; `tools/ref/check_trace.py`) and replays in the page at 144 frames per second, loading in 0.71 s. **Cross-machine:** rank 0 in WSL2 on the desktop (RTX 5070 Ti) dialing rank 1 on `gx10-02` over Tailscale TCP generated for 10.3 minutes with tokens identical to transformers and to its first round, at 0.6 tokens/s, because Tailscale fell back to its NYC relay (31 to 35 ms) instead of a direct path. Logs, load samples, scripts, and both recordings are in `bench/rc/2026-11-16-soak/` (`bench/rc/2026-11-16-soak.md`).

#### [TASK-15b] Multi-Spark RDMA Run (2 to 3 Nodes)
- **Milestone**: RC (Nov 29)
- **Status**: Done
- **Priority**: `could` | **Area**: `benchmarking` | **Tag**: `cut-1`
- **Description**: Only if `gx10-01` or `gx10-02` frees up (cabling is already in place: the three Sparks are direct-cabled in a full mesh): repeat the soak with one engine process per Spark over real node-to-node RDMA.
- **Acceptance Criteria**: Done when a 2 or 3 node run passes the TASK-07 suite across nodes, single-packet p50 latency is within 1.5x of node-to-node `ib_write_lat`, and a 30-minute generation run is archived under `bench/rc/` with scaling against the single-node soak.
- **Results (Nov 11)**: acceptance met with two nodes (`gx10-01` was busy): rank 0 on `gx10-02`, rank 1 on `gx10-03`, RDMA over their direct cable. The TASK-07 suite passes across nodes (20 of 20 dispatch checks, 20 of 20 MoE checks, 45 unit tests on each machine); p50 one-way latency is 4.38 to 4.44 us against `ib_write_lat`'s 3.34 to 3.41 (1.30x, target 1.5x); bandwidth 98.3 Gb/s median against `ib_write_bw`'s 106. A 40-minute generation run went 681 rounds at 54.4 tokens/s, every round bit-identical, memory flat, weights unchanged on both ranks: 2.7x the single-node soak (20.5 tokens/s), mostly from the second GPU. **Open issue:** in an earlier 30-minute repeat, one round (164 of 508) stalled for about 8 seconds and produced different output, with nothing logged; it did not recur, and the soak now reports each differing round and checks the weights so it can be caught. With the expert cache, each rank reads from its own drive and hints add 17 to 23% (`bench/olmoe/2026-11-11-two-sparks.md`). See `bench/rc/2026-11-11-two-sparks.md`.

#### [TASK-16] Telemetry Recording & Trace Replay
- **Milestone**: RC (Nov 29)
- **Status**: Done (built Nov 15)
- **Priority**: `must` | **Area**: `webgl`
- **Description**: The engine writes compact telemetry (routing decisions per token and layer, the process that ran each expert, RDMA and TCP packet timings, expert-cache and NVMe events) using the TASK-04 schema, both to trace files and to TASK-30's live stream (TASK-16b). The 2D view loads a recorded chat session and replays it with the chat text alongside. These recorded sessions are what the public demo ships. Changed Nov 13: TASK-30's 2D view replaces the WebGL inspector as the main view. Recording must not slow the engine:
  - **Only data the engine already has:** routing decisions come from the router on the CPU, packet times from the transport's own timestamps, and expert-cache results from `ExpertCache`, which already counts resident, hinted, late, and missed calls. No GPU synchronization is added to time individual experts.
  - **Off the token path:** each recording thread (the router and the threaded per-token loops) writes fixed-size events to its own ring buffer, sized to stay in L2 (about 64 KB), with head and tail on separate cache lines so threads never contend. A separate thread serializes and writes them out. A full ring drops events instead of blocking, and drops are counted and shown.
  - **Prefill is summarized:** a prefill chunk records counts per layer and expert, not one event per token (a 2,048-token Qwen3 chunk is about 786,000 routing decisions). Decode, which the diagram animates, records every token.
  - **Off unless asked for:** benchmarks run without it.
- **Acceptance Criteria**: Done when a recorded TASK-30 chat session and a TASK-15 run each replay in the 2D view at 60 FPS with correct per-process routing counts (checked against engine counters); a malformed trace file shows an error message instead of crashing the page; and with recording on, tokens per second drop by under 2% and tokens and logits stay identical to a run with it off, in one process and split across two Sparks.
- **Results (Nov 15)**: acceptance met; the TASK-15 run was recorded Nov 16 (60 minutes, 110,979 steps, replayed at 144 frames per second with counts matching the engine; `bench/rc/2026-11-16-soak.md`), and before it a 3-minute soak of the same shape stood in. `olmoe <mode> MODEL --record FILE` writes TASK-16b's per-step frames, one JSON object per line, in every mode (server, long context, golden checks and soak), and under `--serve` places the chat's request, reply text, and finish among the steps (`docs/telemetry-schema.md`). `olmoe.step` now calls optional step hooks on `Moe`, so the tap sees every step on any path; the broadcaster thread writes the file, and a full ring drops steps (counted) instead of blocking. With `tools/ref/check_record.py` (`bench/trace/2026-11-15-recording.md`), alternating runs with `--compare` against one reference: Qwen3-30B-A3B on `gx10-02`, 36.58 tokens/s without recording against 36.65 with it (+0.2%); OLMoE split across `gx10-02` and `gx10-03` over RDMA, 95.95 against 96.10 (+0.2%); tokens and logits bit-identical in all 12 runs, and every recording whole with nothing dropped. A recorded 3-minute soak over RDMA (17,091 steps, 93.3 tokens/s against 92.0 without recording) and two recorded three-turn chats (OLMoE split over RDMA; Qwen3 with reasoning, kept in `bench/trace/2026-11-15/`) each account for every expert call, with per-rank sums equal to the engine's counters. The page replays them (from a file, a drop, or `?trace=URL`, gzipped or not) with play, pause, speed, and scrubbing, the chat alongside, and the engine-counter check; the soak replayed at 144.8 frames per second (7.1 ms per frame at p99) and seeks in 56 ms. Malformed files show why they cannot be replayed; corrupt lines inside a good file are counted and skipped. Departures from the description: packet timings are the per-step round trip the live feed already carries, not individual packets, and the router's single ring stands in for per-thread rings, since only the router thread records.

#### [TASK-16b] Live Telemetry over WebSocket
- **Milestone**: RC (Nov 29)
- **Status**: Done (built Nov 15)
- **Priority**: `must` | **Area**: `webgl`
- **Description**: `olmoe serve` (TASK-30) streams telemetry frames over a WebSocket beside the chat stream, so the 2D view updates as tokens arrive. Plain `ws://` on localhost; `wss://` behind a TLS proxy for a viewer on another machine. Frames may be decoded in JavaScript; a Wasm decoder built from the Zig schema code is optional. Never enabled on the public site. Changed Nov 13: the stream comes from the engine's own server instead of a separate relay, and the task moved from `could` (`cut-1`) to `must`, since the live diagram is half of TASK-30's harness. A slow or disconnected viewer only loses frames (TASK-16's rings drop instead of blocking); it never holds up generation.
- **Acceptance Criteria**: Done when the 2D view shows a live split run with under 500 ms from an expert call to its drawing; malformed frames are rejected without crashing the page; and with a viewer connected, tokens per second drop by under 2% and tokens and logits stay identical to a run with no viewer, in one process and split across two Sparks, including when the viewer stalls (dropped frames are counted, generation is not slowed).
- **Results (Nov 15)**: acceptance met, except that logits were not compared because the server does not return them (tokens were identical in every run). `--serve` streams each engine step's expert counts as JSON frames at `/telemetry` (`src/telemetry.zig`); a wrapper around the engine's `Moe` counts only while a viewer is connected and hands steps to a ring the engine never waits on, and a viewer that falls behind has frames dropped for it (gaps in `seq`). With `tools/ref/check_live.py` (`bench/serve/2026-11-15-live-telemetry.md`): OLMoE split across `gx10-02` and `gx10-03` over RDMA, 105.58 tokens/s with no viewer, 105.35 (-0.2%) with one reading and 105.38 (-0.2%) with one stalled; Qwen3-30B-A3B on `gx10-02`, 38.79 against 38.79 and 38.74. In both, the tokens were the same with and without viewers, the frames accounted for every expert call and matched the engine's own counters, a reading viewer stayed gapless beside a stalled one, which had 854 and 301 frames dropped while generation kept its speed, and frames arrived 16 to 32 ms (median) and at most 155 ms after their step began. The page rejected malformed frames without losing its state.

#### [TASK-17] Capacity Factor & Memory Thrashing Guardrails
- **Milestone**: RC (Nov 29)
- **Status**: Done (built Nov 16)
- **Priority**: `should` | **Area**: `code` | **Tag**: `cut-2`
- **Description**: Enforce per-expert capacity factors, receiver-granted credits, and a hard memory budget per engine process, so load spikes cannot overflow rings or push a shared machine into swapping. Overflow tokens take the residual path and are counted.
- **Acceptance Criteria**: Done when a synthetic 3x load spike completes with dropped-token counts reported, no memory leak (allocator balance at zero after the run), no pipeline stall longer than 100 ms, the process never exceeds its configured memory budget, and token drop rate is under 1% at nominal load.
- **Results (Nov 16)**: acceptance met, except that the first step of 2 of 13 runs waited 336 and 2,250 ms (below). `--capacity C` (off by default, since drops change the output) holds each process to C times its even share of a nominal step's expert calls per layer, dropping the lowest-weight calls past it (residual path) and counting them; rank 1 grants its limit to rank 0 as a control immediate (`docs/wire-format.md`), and the ring's slot credits still keep rings from overflowing. `--expert-capacity C` is the per-expert form, but OLMoE's routing concentrates so hard in a 64-token chunk that it needs factor 7 for under 1% drops, against 1.25 per process (`bench/rc/2026-11-16-guardrails/capacity_curve.py`). `--memory-budget MB` puts every allocation, now including the KV cache, through a counting heap (`src/budget.zig`) that refuses to grow past what the weights and GPU leave, and checks resident plus GPU memory after every step; every process reports its heap balance at exit. `--spike F` brings the middle third of a `--context` prompt in steps F times as big. OLMoE split across `gx10-02` and `gx10-03` over RDMA, 4,032 tokens in 64-token steps with a 3x spike: with `--capacity 1.25 --memory-budget 7000` (4 runs) 0.18% of calls were dropped before the spike and 0.22% after, spike steps took at most 700 ms instead of 1.7 s, memory stayed at most 5,170 and 3,545 MB with no check over budget, rank 0 never waited on rank 1 for more than 2.3 ms, and both heaps ended with 0 bytes outstanding; budgets too small for the KV cache or the weights were refused instead of exceeded. Without the guardrails (7 runs), no wait during or after the spike passed 1.2 ms, but the first step of two runs waited 2,250 and 336 ms, with no packet resent (NIC counters 0) and rank 1's slowest call at 4.3 ms, so most likely a cold start on the shared `gx10-03`. Qwen3-30B-A3B on `gx10-02` dropped nothing at nominal load and stayed within a 40,000 MB budget. Without the new options, tokens and logits are bit-identical to the committed build and decode speed is within 0.3% (`bench/rc/2026-11-16-guardrails.md`).

#### [TASK-30] OpenAI-Compatible Server & Built-in Chat Harness
- **Milestone**: RC (Nov 29)
- **Status**: Done (built Nov 15)
- **Priority**: `must` | **Area**: `code`
- **Description**: Added Nov 13, after Testing Session 1 (`bench/playtest/session-1.md`): testers praised the polish, but most could not tell what the simulated mission showed, and a simulation undersells a working engine. What sells it is using a real model through it.
  - **Server:** `olmoe serve` runs the router (rank 0, in one process or split over any transport, with the tracker) as an HTTP server: `POST /v1/chat/completions` in OpenAI's format, with server-sent-event streaming, and `GET /v1/models`. `curl`, the `openai` Python package, and existing chat clients work unchanged.
  - **Built-in harness:** the same server serves the page from `web/`: a chat panel beside a 2D live diagram showing the machines with their expert grids, each token's chosen experts lit on the machine that ran them, links with live latency and packet counts, expert-cache hits and misses, and tokens per second. It is fed by TASK-16b's stream; sessions can be recorded for replay (TASK-16). Expert-cache hits and misses only happen when experts stream from the drive (`--cache N`); with every expert loaded, the default, every call is a hit. So one demo setup runs with `--cache` on purpose, where the lookahead's prefetches visibly matter.
  - **Tokenizer:** a byte-level BPE tokenizer read from each model's GGUF metadata or its `tokenizer.json` (OLMoE-1B-7B-Instruct, Qwen3-30B-A3B), their chat templates, and streamed decoding. Generation reuses TASK-24's loop, and TASK-28's saved prefixes skip repeated system prompts.
  - **Not in scope:** third-party harnesses (agent CLIs, desktop chat apps) are not dependencies; they connect through the standard API if they take a custom base URL. An Anthropic-format endpoint for agent tools is a stretch demo. There is no authentication: the server binds to localhost by default and is only ever exposed on a private network; public access belongs to TASK-25.
- **Acceptance Criteria**: Done when (1) the engine's tokenizer gives the same token ids as Hugging Face's `tokenizers` on the TASK-24 prompts and a long text, for both models; (2) greedy chat requests through `curl` and the `openai` Python package, streamed and not, return the same tokens as `olmoe --generate` for the same prompt tokens; (3) the built-in page chats with OLMoE-1B-7B split across two Sparks over RDMA and with Qwen3-30B-A3B on one, streaming tokens while the diagram updates live, with its per-machine expert counts matching the engine's counters; and (4) time to first token and tokens per second through the server, against `olmoe` directly and with the diagram connected or not, are recorded in `bench/`.
- **Progress (Nov 15)**: criterion (1) met. `src/tokenizer.zig` gives the same ids as Hugging Face's `tokenizers` for both models, from the GGUF and from `tokenizer.json`, on the golden prompts, 22 edge cases, 500 seeded random strings, and a 474k-token text (`tools/ref/check_tokenizer.py`). `src/chat.zig` renders conversations exactly as transformers' `apply_chat_template` does (72 of 72 prompts per model, `tools/ref/check_chat.py`), including the older Qwen3 template our GGUF carries. Criterion (2) met the same day: `olmoe <mode> MODEL --serve [IP:]PORT` (`src/serve.zig`, `src/completion.zig`) answers `/v1/chat/completions` (streamed or not, with Qwen3's reasoning as `reasoning_content`) and `/v1/models` in every mode, reusing the previous request's shared prompt prefix unless `--no-reuse`. With `--no-reuse`, greedy replies through `curl` and the `openai` package, streamed and not, gave the same tokens as `olmoe local --context N --tokens F` for the same prompt tokens (`tools/ref/check_serve.py`): OLMoE on the desktop CPU (6 tokens/s), Qwen3-30B-A3B on `gx10-02`'s GPU (39 tokens/s, first token 130 ms), and OLMoE split over RDMA on `gx10-02` (35 tokens/s; identical with `--payload f32`, while the default FP8 payload diverges after 11 to 25 tokens in two of four replies). Criterion (3) met the same day: the server serves the page at `/` (`web/live.html`, embedded in the binary), a chat panel beside a 2D diagram with one grid of layers by experts per rank, each token's experts lit on the rank that ran them, the link's packets per second and round trip per layer, and each rank's expert calls checked against the engine's own counters. It chatted with OLMoE split across `gx10-02` and `gx10-03` over RDMA (103 tokens/s) and with Qwen3-30B-A3B on `gx10-02` (38 tokens/s, reasoning shown apart), with the page's counts matching the engine's for every rank, at 144 frames per second in the browser (TASK-16b; `bench/serve/2026-11-15-live-telemetry.md`). Criterion (4) met the same day (`bench/serve/2026-11-15-server-latency.md`, `tools/ref/bench_serve.py`): on the same prompt tokens (14 to 1,561 of them, 128 generated), the server's time to first token and decode speed match `olmoe` run directly (decode within 1.5%, first token within 10% and never slower), for OLMoE split across `gx10-02` and `gx10-03` over RDMA (107 ms and 105 tokens/s on a short prompt) and Qwen3-30B-A3B on `gx10-02` (125 ms and 38.7 tokens/s). The client sees the first token 1 ms after the engine does, or two decode steps later for Qwen3, whose replies open with `<think>` and a newline that show no text. A viewer on the live feed changed neither, and the tokens were identical in every run. The split router now also runs `--context N --generate M` directly, as `local` did.

---

### Milestone 5: Final Release (Target: Dec 6)

#### [TASK-18] GitHub Pages Showcase Deployment
- **Milestone**: Final (Dec 6)
- **Status**: Done (deployed Nov 16)
- **Priority**: `must` | **Area**: `docs`
- **Description**: Host TASK-30's 2D view at https://aiforgames2026.github.io/DistributedMoEEngine/, replaying recorded chat sessions and RC runs (TASK-16), each labeled as a recording of the real engine. The 3D inspector and the guided mission may stay as a secondary tab. Static assets only: HTML, JS, optional Wasm, and trace data; the public site never connects to an engine. Changed Nov 13: the 2D view replaces the WebGL inspector as the main page.
- **Acceptance Criteria**: Done when the public URL loads the 2D view with at least one recorded chat session in current Chrome, Firefox, and Safari without any binary download or install prompt, a CI check fails the deploy if the bundle contains `.exe`, `.dmg`, `.pkg`, `.msi`, or shell installer files, a Content-Security-Policy is set, and initial load is under 5 MB.
- **Results (Nov 16)**: acceptance met. Deployed by commit `9377196`; the Pages workflow and CI both passed. The site's main page is now the engine's own `web/live.html` in a showcase mode, which `web/scripts/showcase.mjs` turns on at build time. In that mode the page never connects to an engine or posts a chat. It offers four recordings from `web/showcase.json`, each badged "Recording of the real engine" and labeled with its date, machines, and a link to its raw results: the OLMoE chat split over RDMA and the Qwen3-30B-A3B chat (TASK-16), the first 6,000 steps (3.3 minutes) of TASK-15's one-hour soak (`bench/rc/2026-11-16-soak/soak-first-6000.jsonl.gz`), and TASK-15's cross-machine run. The 3D inspector and its guided mission moved to `inspector.html`, linked both ways. The built page carries its script and styles as files, so its Content-Security-Policy allows nothing inline and no connections beyond the site. `web/scripts/check-bundle.mjs` runs in `npm run build` and in the Pages workflow. It fails the deploy for an executable or installer, found by extension or by its first bytes, for a page without a CSP, or for a main page over 5 MB to load; its self-test, in `npm test`, plants each of these. The main page loads 0.125 MB before playing. Every recording replayed with its per-rank counts matching the engine's, and with no console errors, in the Claude browser pane (Chromium) and in Firefox 159 (`web/scripts/check-firefox.mjs`, over WebDriver BiDi). On the public URL, the deploy's bundle check passed (0.125 MB initial load, 25 MB bundle including the PoC video), the page carries its CSP, and the same Firefox check passed for all four recordings, with counts matching. The Claude browser pane (Chromium) replayed the default chat with no console errors. Safari on an iPhone 12 Pro, checked by hand by the project owner, replayed the recordings and worked.

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

### Post-v1 Roadmap (no milestone)

#### [TASK-25] Operator Dashboard
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 17)
- **Priority**: `could` | **Area**: `webgl`
- **Description**: Added Nov 5 from the project owner's direction: the web page grows into a dashboard where an operator manages nodes, the router, and models, and the API endpoint at the router that harnesses call. The inspector becomes one feature of it. This needs what v1 does not ship: the tracker (TASK-09), live telemetry (TASK-16b), and an authenticated control channel on top of TASK-30's server. Testing Session 1's question 5 asks which control to build first. Until this task starts, the public site stays simulated and recorded data only.
  - **Design (project owner's choices, Nov 17):** a supervisor plus node agents. `olmoe dashboard` runs on rank 0's machine, serves the page and the API, starts the router as its own child, and passes `/v1` through to it; each other machine runs `olmoe node`, an agent that starts and stops one expert rank on the dashboard's command, so no engine internals change. A machine that joins is used from the next load; moving experts onto it while a model runs stays out. Operators sign in with an admin token kept in a file and get a session cookie; agents prove a separate shared secret. The agents link to the dashboard directly instead of through TASK-09's tracker, which has no authentication.
  - **Not in scope:** TLS (a proxy in front, or a private network), user accounts, tool calls, and the public site, which stays recordings only.
- **Acceptance Criteria**: Written when scheduled (Nov 17).
  - **Operator flow:** from the page, an operator signs in, approves a machine whose agent has joined, loads OLMoE-1B-7B across it and rank 0, chats with it through the dashboard's `/v1` with the live diagram showing, and stops it, leaving no engine process running. Greedy replies through the dashboard's `/v1` (with `--no-reuse`) give the same tokens as `olmoe local` on the same prompt tokens (`tools/ref/check_serve.py`).
  - **Authentication:** every control action and the live view answer 401 without a session, with a wrong token, and with a forged or signed-out session; a cross-site POST is refused even with the token; an agent without the node secret, or a frame with a bad signature, is refused (`tools/ref/check_dashboard.py`, plus unit tests in `zig build test`).
  - **Failures:** a rank that dies or a node that drops while a model runs marks the run failed within 2 s and stops the rest, with the reason on the page.
  - **Across machines:** the operator flow with the router on a Spark and an expert rank on a second machine.
- **Results (Nov 17)**: acceptance met, over TCP (`bench/serve/2026-11-17-dashboard.md`); RDMA through the dashboard is untested. Across machines: the dashboard and router on `gx10-02` with the desktop (WSL, behind NAT, so the new `--advertise` and an SSH tunnel) as an expert node over Tailscale: from the page, sign-in, approval, a two-process load and a streamed chat; through `/v1` the same 16 tokens as `olmoe local` on 4 of 4 conversations; `check_dashboard.py` 16 of 16; a killed desktop rank marked the run failed 0.3 s later and stopped the router. On one machine first: `olmoe dashboard` (`src/dashboard.zig`, page `web/dashboard.html` and `web/dashboard.js`) and `olmoe node` (`src/node_agent.zig`), with sessions and the signed node link in `src/auth.zig` and child processes in `src/proc.zig`. On the desktop (WSL, CPU, Debug build), with two agents beside the dashboard: from the page, sign-in, approval (kept across restarts), a load of OLMoE-1B-7B over three processes over TCP (ready in about 4 s, live diagram shown), a streamed chat through `/v1`, reload, and stop leaving no engine process. Through the dashboard's `/v1`, curl and the `openai` package, streamed and not, agreed on 4 of 4 conversations, and `olmoe local` generated the same 32 tokens for all 4. `tools/ref/check_dashboard.py` passed 16 of 16 (401 for every control action without a valid session or token, 403 for a cross-site POST, cookie HttpOnly and SameSite=Strict and dead after sign-out, an agent with the wrong secret or a badly signed frame refused). Killing a rank marked the run failed and stopped the rest within 1 s. `zig build test` passes 114 tests.

#### [TASK-27] Compact KV Cache Formats
- **Milestone**: None (after v1)
- **Status**: Done (built Nov 16, ahead of v1)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 10, after DeepSeek-V4.1-Flash's KV cache work (arXiv 2609.19969) came up in class. Its 890 bytes per token come from architecture and training (shared KV across layers, sparse top-k selection, FP4 cache with quantization-aware training), none of which an existing model like OLMoE can take on without retraining. The part that carries over is storing the cache smaller: TASK-24's cache holds K and V in f32 (262,144 bytes per token for OLMoE-1B-7B), and a BF16 or FP8 (E4M3 with per-block scales) option would cut that by 2x to 4x. The cache lives on the attention rank only, so this saves memory there and changes no wire traffic.
- **Acceptance Criteria**: Written when scheduled (Nov 16): `--kv-format bf16` and `--kv-format fp8` work on the CPU and the GPU, with saved prefixes (TASK-28) and with the NVMe tier (TASK-12), each giving exactly the output of an all-in-memory cold run of the same format. f32 stays the default and bit-identical to the committed build, at the same speed. Each format's bytes per token, its logits error and matching-token count against transformers' `generate()` on the TASK-24 prompts (OLMoE-1B-7B and Qwen3-30B-A3B), and decode speed against f32 at 32k and 128k, recorded in `bench/`.
- **Results (Nov 16)**: acceptance met (`bench/olmoe/2026-11-16-kv-formats.md`, `src/kv_format.zig`).
  - **Formats:** bf16 rounds to nearest even. fp8 stores E4M3 with a power-of-two scale per KV head of each position, one byte after the row. Rows are encoded once, as they enter the cache, and the CPU and every GPU kernel decode exactly the same numbers. Each GPU attention kernel is compiled once per format; choosing the format at run time had slowed f32 prefill attention by 32%.
  - **Size:** Qwen3-30B-A3B at 128k takes 25.77 GB in f32, 12.88 GB in bf16, and 6.64 GB in fp8; OLMoE-1B-7B's 262,144 bytes per token become 131,072 and 66,048.
  - **Against transformers:** bf16 keeps every OLMoE top-1 and all 192 generated tokens (worst logits error 1.45), and Qwen3's 90 of 96 tokens. fp8 is lossier: 66 of 192 OLMoE tokens and 60 of 96 Qwen3 tokens, with 38 of 43 OLMoE prompt top-1s. A temporary patch rounding keys and values separately showed the loss is in the keys. Keys in bf16 with values in fp8 would likely keep bf16's agreement at 2.67x smaller.
  - **Speed on `gx10-02`, Qwen3 with YaRN 4:** decode at 32k went from 15.5 tokens/s to 22.0 (bf16) and 26.6 (fp8), and at 128k from 5.56 to 9.63 and 14.43, with prefill unchanged or faster. bf16 matched f32's 128 tokens at 32k; at 128k both compact formats picked a different first token.
  - **Exactness:** saved prefixes restore exactly in both formats, at half and a quarter the file size. With the middle on NVMe the output is exactly the all-in-memory run's. f32 is bit-identical to the committed build at 32k, at the same speed.
  - **Also changed:** a decode-kernel bug, where value rows past the last position could turn into NaN, was fixed for every format. `--compare` now reports how many tokens match before the first difference.
  - **Follow-up, `bf16_fp8` (Nov 16):** keys in bf16 and values in fp8, 2.64x smaller than f32 on Qwen3 (9.76 GB at 128k). It keeps bf16's prompt top-1 agreement (43 of 43 OLMoE, 36 of 37 Qwen3), with 146 of 192 OLMoE and 72 of 96 Qwen3 generated tokens equal to `generate()`. Decode at 32k is 24.2 tokens/s (bf16 22.0, fp8 26.4) and at 128k 11.62 (f32 5.56). Saved prefixes and the NVMe tier stay exact, and f32 stays bit-identical (`bench/olmoe/2026-11-16-kv-bf16-fp8.md`).

#### [TASK-28] Persistent Prefix KV Cache
- **Milestone**: None (after v1)
- **Status**: Done (built Nov 11, ahead of v1)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 10, from the same paper, which keeps every prompt's global KV on SSD for at least 72 hours so a repeated prefix skips prefill. Save the K and V of a prompt prefix to NVMe, keyed by a hash of its tokens, and read them back through the TASK-03 `io_uring` reader when a later prompt starts with the same tokens. OLMoE has no sliding-window cache, so unlike DeepSeek's bounded replay the restored cache is exact. Shares the NVMe paging code with TASK-12 if that task is built.
- **Acceptance Criteria**: Written when scheduled (Nov 11): a prompt that reuses a saved prefix, wholly and in part, produces exactly the same tokens and logits as a cold run, at 16k and at 128k on Qwen3-30B-A3B, with time to first token for both recorded in `bench/`.
- **Results (Nov 11)**: acceptance met on `gx10-02` (Qwen3-30B-A3B, `--gpu --yarn 4`). `olmoe --context N --prefix-cache DIR` saves the prompt's KV cache, tokens, and last logits to a file after prefill (O_DIRECT, 5.9 GB/s) and restores the longest matching saved prefix through the io_uring reader (6.1 GB/s). Time to first token: 509.7 s cold against 4.2 s restored at 128k (25.8 GB read), and 17.1 s against 0.6 s at 16k. A 16k prompt sharing 12,000 tokens with a saved one took 8.4 s. Every restored run matched its cold run bit for bit (tokens and logits), because a partial match takes only whole chunks and keeps a cold run's chunk boundaries, and a file is used only when its fingerprint (engine build, model, settings that change rounding) matches. In-memory KV cache only for now. See `bench/olmoe/2026-11-11-prefix-cache.md`.

#### [TASK-29] Qwen3.6-35B-A3B (Hybrid Linear Attention)
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 16, ahead of v1)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 11, when choosing a long-context model for TASK-12. Qwen3.6-35B-A3B (April 2026, 35B total, 3B active, 262,144-token native context) fits the lab: its Q8_0 GGUF is 36.9 GB and its BF16 checkpoint 72 GB. But it is a hybrid. Of its 40 layers, 30 are Gated DeltaNet (linear attention with a fixed-size recurrent state) and 10 are full attention with an output gate, 2 KV heads of 256 dims, and partial RoPE (64 of 256 dims). It has 256 experts, 8 routed plus 1 shared per token. Supporting it means a Gated DeltaNet layer (CPU and a CUDA kernel for long prefill), gated attention, partial RoPE, and shared experts on top of TASK-12's Qwen3-MoE work. The MTP head and vision encoder stay out. TASK-12 used Qwen3-30B-A3B instead, because this model's KV cache is small (about 5 GB at 128k tokens in f32) and would not exercise NVMe offload.
- **Acceptance Criteria**: Written when scheduled (Nov 16), on ggml-org's Q8_0 GGUF against transformers' `Qwen3_5MoeForCausalLM` (float32, on the GGUF's dequantized weights): the TASK-10 checks (logits within 1e-2, top-1 identical, every routing choice identical) and TASK-24's greedy generation check, in one process and split across two (experts on rank 1). A tiny random model with the same layer mix checks the same in `zig build test`. The Gated DeltaNet layers and the 256-dim gated attention run on the GPU, so 32k and 128k prompts complete on `gx10-02`, with time to first token and decode speed recorded in `bench/`. Saving the recurrent state with `--prefix-cache` is left for later.
- **Results (Nov 16)**: acceptance met (`bench/olmoe/2026-11-16-qwen35.md`).
  - **Against transformers:** logits within 5.4e-5 (target 1e-2), every top-1 and all 12,480 routing choices identical, and all 96 generated tokens equal to `generate()`, on the CPU, on the GPU, and split across two processes over shared memory and RDMA (`--payload f32`). One generated step's logits differ by 0.335: there transformers' router has an exact tie, and its own full forward pass differs from its `generate()` by the same 0.335.
  - **Built:** the Gated DeltaNet layer (CPU, and a GPU delta-rule kernel), gated attention with partial rope and a 256-dim GPU attention kernel, the shared expert, a KV cache that holds the 10 attention layers plus each Gated DeltaNet layer's fixed state, the `qwen35` tokenizer (identical to Hugging Face on 528 texts), and a tiny test model checked in `zig build test`.
  - **Long context on `gx10-02`:** first token in 109 s at 32k, 1,253 s at 128k, and 4,845 s at the native 262k, decode at 24.6, 13.7 and 8.7 tokens/s. The KV cache takes 40,960 bytes per token, so decode at 128k is 2.5 times Qwen3-30B-A3B's. Prefill was bound by the 256-dim attention kernel, plain f32 at about 1.3 TFLOPS.
  - **Tensor-core attention (Nov 16):** a 256-dim tensor-core kernel (about 11 TFLOPS, within 5.7e-6 of the f32 kernel) brings the first token to 52 s at 32k, 305 s at 128k and 890 s at 262k, and decode to 26.4, 16.1 and 10.6 tokens/s, with the golden check unchanged (`bench/olmoe/2026-11-16-tc256.md`).
  - **Not yet:** `--kv-offload` and `--prefix-cache` refuse this model, since neither saves the recurrent state, and the server's prompt reuse starts a hybrid model's session over.
  - **Golden regenerated (Nov 18):** its generation check had failed since TASK-31's near-tie rule, which needs each step's router gap in the golden, and this one predated it. `ref_olmoe.record` can now record the gaps (`gaps=True`, on for `ref_qwen35.py`); the regenerated `qwen35_generate.tfref` is identical to the old one apart from them, and the check passes with step 3's exact tie (gap 0) excused (`bench/olmoe/2026-11-16-qwen35.md`).
- **Track (Nov 15)**: first step of the hybrid linear-attention track, TASK-29 then TASK-31 then TASK-32. The Gated DeltaNet layer, its saved recurrent state (in place of TASK-28's saved KV for those layers), and the scheduling of mixed layer types are built here, at a size the lab can check against transformers, and reused by both later models.

#### [TASK-31] Qwen3.8-Flash-Next (Hybrid Attention, Expert Streaming at Scale)
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 16 to 17, ahead of v1)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 15, second step of the hybrid track. As reported in summaries of its paper (to be checked against its `config.json` before scheduling): 125B parameters with 6B active per token, plus 51B of n-gram embedding tables kept off the GPU and a 4B multi-token prediction module; 512 experts, 10 routed plus 1 shared per token; 48 layers, three Gated DeltaNet layers to every Qwen Sparse Attention (QSA) layer, which attends to selected blocks of tokens. At 4 bits the 125B core is roughly 65 to 70 GB, so it fits on two Sparks, or on one with experts streamed from NVMe (`--cache`). Its small experts (roughly 2 to 3 MB at 4 bits) stress the expert cache and lookahead (TASK-11, TASK-14) harder than any model so far, and its n-gram tables are a per-token lookup that could also stream from NVMe. New work on top of TASK-29: QSA, the n-gram tables, and whichever weight format its releases use (K-quant GGUFs need a new loader path; NVFP4 or Q8_0 would not). The MTP module stays out.
- **Checked against the release (Nov 16)**: `config.json` (`qwen4_exp`), ggml-org's GGUF (`qwen4exp`, Q8_0, 163 GB in two parts) and transformers 5.23's `modeling_qwen4_exp.py`.
  - **Layers:** 48, hidden size 2,560, with three Gated DeltaNet layers (16 key heads, 48 value heads of 128, sigmoid output gate) to every QSA layer (24 query heads, 2 KV heads of 256, partial rope over 64 dims, an output gate).
  - **QSA:** an indexer (4 heads of 128, one shared key) scores blocks of 4 tokens by their mean key and keeps the best 512 blocks (2,048 tokens), plus the unfinished last block. Below 2,048 tokens of context it is dense attention.
  - **Hyper-connections:** the residual is 4 streams of 2,560. Before each attention and MoE block, a low-rank gate (rank 320) mixes the streams into one input, and the block's output is added back to each stream with its own weight. A final mixer replaces the output norm.
  - **N-gram embedding:** at layer 1, each token's bigram and trigram hash into 16 tables of about 20 million rows. The rows are 160 values each (28.8 GB in Q4_0), gated per stream and passed through a short dilated convolution.
  - **MoE:** 512 experts of 640 (2.5 MB each in Q8_0, 128 GB in all), 10 routed plus 1 gated shared expert, softmax router.
  - **Weights:** Q8_0, plus F32 norms, BF16 indexer projections and the Q4_0 n-gram table. The MTP head, the drafter and the vision encoder stay out.
- **Acceptance Criteria** (Nov 16, following TASK-29):
  - **Against transformers:** on ggml-org's Q8_0 GGUF, compared with transformers' `Qwen4ExpForCausalLM` in float32 on the GGUF's dequantized weights. A reference script loads experts and n-gram rows only when used, since the whole model in float32 would not fit in memory. The checks are TASK-10's (logits within 1e-2, top-1 identical, every routing choice identical) and TASK-24's greedy generation check.
    - These run on short prompts, where every selected QSA block must match too, and on one longer than 2,048 tokens, so QSA's block selection is exercised.
    - **The long prompt (amended Nov 16):** past QSA's budget, choosing 512 of some 650 blocks per query amplifies float rounding through the layers, so no implementation reproduces transformers' choices exactly. transformers rerun with 7 threads instead of 20 shares 99.912% of its own QSA blocks and 99.64% of its expert choices, with logits 1.96 apart. So the long prompt needs: the first QSA layer's choices identical, at least 99.9% of QSA blocks and 99.5% of expert choices shared, and every top-1 token identical. The 1e-2 logit bound applies to the short prompts.
    - They run in one process, and split across two (experts on rank 1).
  - **Unit test:** a tiny random model with the same layer mix, a QSA budget small enough to select, hyper-connections and the n-gram embedding checks the same in `zig build test`.
  - **GPU and memory:** the new layers run on the GPU. The model runs on `gx10-02` alone, with experts streamed from NVMe (`--cache`). 32k and 128k prompts complete, with time to first token, decode speed and expert-cache hit rates recorded in `bench/`.
  - **Chat replay:** a recorded chat with the model (the same scripted three-turn chat as the other replays) is added to the public page's list of recordings, with its setup and speeds written up in `bench/trace/`.
  - **Later:** a split across two Sparks with every expert in memory waits for a second free Spark. Saving the recurrent state with `--prefix-cache` stays out, as in TASK-29.
- **Results (Nov 17)**: acceptance met on `gx10-02` alone, experts streamed from NVMe through a 12,000-slot cache (`bench/olmoe/2026-11-17-qwen38.md`).
  - **Against transformers:** the three short prompts' logits are within 9.73e-5, and all 18,720 routing choices are identical.
    - The 2,600-token prompt shares 99.928% of QSA blocks and 99.698% of expert choices. Its first QSA layer is identical.
    - Every top-1 and all 128 generated tokens equal `generate()`.
    - This holds in one process on the GPU and split across two over shared memory (`--payload f32`), with identical numbers in both.
    - Two router near-ties in transformers (gaps of 8.4e-9 and 3.6e-8) end the logit check at that step. The tokens still match.
  - **Built:**
    - hyper-connections, the n-gram embedding (Q4_0) and QSA, with three GPU kernels for QSA;
    - split GGUF loading;
    - lookahead through the hyper-connections;
    - `--kv-offload` for hybrid models;
    - the `qwen38` chat format with `reasoning_effort` (108 of 108 prompts identical to transformers);
    - a tiny test model in `zig build test`.
    - The tokenizer is identical to Hugging Face on 528 texts.
  - **Long context:**
    - 32k: first token in 1,211 s, decode at 5.71 tokens/s.
    - 128k: first token in 4,971 s, decode at 4.22 tokens/s, peak memory 104 GB.
    - Experts set the pace: 90 to 91% of calls hit the cache, and the 128k prefill read 6.8 TB from NVMe.
  - **Chat replay:** on the public page (`bench/trace/2026-11-17-qwen38-chat.md`), at 7.0 to 8.4 tokens/s. The live diagram wraps a layer's experts onto several rows when there are too many for one.
  - **Unexplained:** two early GPU runs, out of about 13, produced a non-finite value in the long prompt's prefill.
    - `step()` now stops on one and names the layer.
    - Every expert call is checked and rechecked on the CPU if bad.
    - 20 traced runs and every later run were clean.
  - **Not yet:** `--prefix-cache` and the server's prompt reuse still start a hybrid model over. The two-Spark in-memory split waits for a free Spark.

#### [TASK-32] GLM-5.3-Flash (Hybrid Attention, 320B)
- **Milestone**: None (after v1)
- **Status**: In Progress (started Nov 17)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 15, third step of the hybrid track. As reported by NVIDIA's model coverage page and hosting providers (to be checked against its `config.json`): 320B parameters with 18B active per token; 45 layers, the first 3 with dense feed-forward blocks and the other 42 with 288 experts, 8 routed plus 1 shared per token; 34 KDA layers (a delta-rule linear attention with per-channel gates, which generalizes TASK-29's Gated DeltaNet) and 11 sparse MLA-style attention layers with a token indexer. Its smallest 4-bit GGUF is about 200 GB, so it needs two Sparks in memory or one streaming experts from NVMe. New work on top of TASK-31: per-channel gating in the delta-rule kernel, MLA with sparse token selection, dense leading layers, and the router (likely sigmoid with a bias, to be confirmed). The vision encoder stays out. Generating transformers references for a 320B model may need the lazy expert loading TASK-12's reference script uses, or more memory than the lab has; settle that before scheduling.
- **Checked against the release (Nov 17)**: `config.json` (`glm5_next`), ggml-org's GGUF (`glm5-next`, Q4_K, 181 GB in two parts) and transformers 5.23's `modeling_glm5_next.py`.
  - **Layers:** 45, hidden size 4,096. Three KDA layers to every MLA layer (34 and 11). The first 3 layers have a dense feed-forward block (12,288), the other 42 an MoE block.
  - **KDA:** Gated DeltaNet with a decay per key channel. 64 heads of 128, separate short convolutions for q, k and v, a low-rank forget gate bounded below at -5, a sigmoid beta, and an output norm gated through a low-rank projection.
  - **MLA with DSA:** 64 heads, queries through a 1,536 rank, keys and values through a 512-wide latent, heads of 256, no rope at all. An indexer (32 heads of 128) scores pools of 4 tokens, each pooled by a learned softmax over its tokens, and keeps the best 512 pools (2,048 tokens) plus the unfinished tail, as in TASK-31's QSA.
  - **mHC:** 4 residual streams. Before each attention and feed-forward block, a projection of all four gives the input mix, the output weights and a 4 x 4 stream mix made doubly stochastic by 20 Sinkhorn iterations. The streams are averaged before the final norm.
  - **MoE:** 288 experts of 2,048 (14.2 MB each in Q4_K, 171 GB in all), 8 routed plus 1 shared, a sigmoid router with a bias used only for choosing, weights renormalized and scaled by 2.5. SwiGLU inputs are clamped at 10.
  - **Weights:** routed experts in Q4_K, everything else Q8_0 or F32. The MTP layer and the vision encoder stay out. Tokenizer: the `glm4` pre-tokenizer.
- **Acceptance Criteria** (Nov 17, following TASK-31; the user chose a multi-machine run in place of one Spark streaming from NVMe):
  - **Against transformers:** on ggml-org's Q4_K GGUF, compared with transformers' `Glm5NextForConditionalGeneration` (text only) in float32 on the GGUF's dequantized weights, loaded lazily. The TASK-10 checks (logits within 1e-2, top-1 identical, every routing choice and every DSA pool identical) and TASK-24's greedy generation check on short prompts, and TASK-31's long-prompt criterion on one past 2,048 tokens (the first DSA layer identical, at least 99.9% of DSA pools and 99.5% of expert choices shared, every top-1 identical). Router near-ties end the logit check at that step, as in TASK-31. These run in one process and split across processes.
  - **Amended Nov 18** (user's decision, after transformers with 1e-6 noise on its embeddings shared only 97.943% of expert choices and 99.795% of DSA pools with its plain run; `bench/olmoe/2026-11-17-glm53.md`):
    - **Long prompt:** at least 97% of expert choices and 99.6% of DSA pools shared, the first DSA layer identical, every top-1 identical except where transformers' own top two logits are less than 0.1 apart (added Nov 18, user's decision, after the cache fix below moved position 2598, 0.005 apart). Its generated tokens are reported, not held.
    - **Short prompts:** when the first layer with a flipped expert choice flipped where the engine's own gap between its 8th and 9th expert is below 1e-4, that prompt's logits (forward and generated) are held to its tokens only.
  - **Unit test:** a tiny random model with the same layer mix, a DSA budget small enough to select, mHC, KDA and the dense layers checks the same in `zig build test`.
  - **Multi-machine run:** the three Sparks over RDMA, every expert in memory (about 57 GB per Spark), in a short window when the other Sparks' servers can be stopped. The window is scripted: the owners' servers are stopped, gx10-01's watchdog paused, and both restored and checked healthy before it ends. It records the golden check, 32k and 128k prompts (time to first token, decode speed, calls per rank), and the chat replay.
  - **GPU:** the new layers and Q4_K experts run on the GPU.
  - **Tokenizer and chat format:** identical to Hugging Face, as in TASK-31.
  - **Chat replay:** the scripted three-turn chat from the three-Spark run is added to the public page, written up in `bench/trace/`.
- **Progress (Nov 18)**: the real-model golden passes the amended criteria (one process, `golden-amended.log`); the investigation is in `bench/olmoe/2026-11-17-glm53.md`.
  - **Golden:** short prompts 0 and 2 pass (logits 2.2e-3); prompt 1 fails on 2 router flips; the long prompt has every top-1 identical but shares 97.24% of expert choices and 99.71% of DSA pools. A two-Spark split gives exactly the same.
  - **Narrowed:** not the GPU kernels, the experts or DSA selection. The difference grows from layer 3, in mHC's stream mixing, which amplifies small input differences about 16 times; the step itself matches transformers in float64.
  - **Sensitivity:** transformers with 1e-6 noise on its embeddings shares 97.943% of expert choices and 99.795% of DSA pools with its plain run on the long prompt (engine: 97.242% and 99.708%), so the long-prompt targets were beyond any f32 implementation. The criteria are amended above.
  - **Multi-machine:** `gx10-03` was in use, so the run moved to two Sparks (`gx10-02` and `gx10-01`). The first window lost `gx10-01`'s watchdog lock to a quoting bug and caused an OOM there (fixed); chat, 32k and 128k wait for a rerun.

#### [TASK-33] Any Number of Expert Nodes
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 16)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 16, before TASK-31. Rank 0 has always talked to exactly one expert process, so a model larger than two machines' free memory could not run in memory. TASK-31's Q8_0 GGUF (163 GB) needs about 115 GB on `gx10-02` plus more elsewhere: the desktop's free RAM, and whatever `gx10-01` and `gx10-03` have left once their owners agree. This task lets rank 0 route to any number of expert processes:
  - Each peer is reached over TCP or RDMA, chosen per peer, and gets its own flow control, health check, capacity grant (TASK-17) and prefetch hints (TASK-11).
  - The placement manifest covers any number of ranks, with an even split by default. `shm` and `rdma` take `--ranks N` to fork N-1 local expert processes.
  - With the tracker (TASK-09), a router given no peers waits for `--ranks N` - 1 healthy experts (one by default, as before), takes each expert from the first of them that announced it (by announced rank, then tracker id), holds any expert none of them owns itself, and refuses to route once any of them turns unhealthy, as it does today with one. Moving a lost peer's experts elsewhere stays out.
  - Recordings, the live diagram and the replay page (TASK-16, TASK-16b) show each machine's calls.
- **Acceptance Criteria**:
  - **Bit for bit:** on one machine with CPU experts, runs over 3 and 4 processes give logits and generated tokens bit-identical to the two-process run.
  - **Goldens:** OLMoE and Qwen3-30B-A3B pass the TASK-10 and TASK-24 checks over shared memory with `--ranks 3` and `--ranks 4`, and over TCP across `gx10-02`, the desktop and a third process.
  - **Tracker:** a tracked run gathers 3 experts by itself and refuses once one is stopped.
  - **Unit test:** a 3-rank split runs in `zig build test`.
- **Results (Nov 16)**: acceptance met (`bench/olmoe/2026-11-16-nodes.md`).
  - **Bit for bit:** with CPU experts, Qwen3-30B-A3B over 2, 3 and 4 processes gives the 16 generated tokens and their logits of one process, bit for bit.
  - **Goldens:** OLMoE and Qwen3-30B-A3B (plain and with YaRN) pass over shared memory with 3 and 4 processes, OLMoE over RDMA with 3 and 4, and all three over TCP with rank 0 and rank 1 on `gx10-02` and rank 2 on the desktop's CPU in WSL2.
  - **Tracker:** a router given only `--tracker --ranks 4` gathered 3 experts and passed. A second run refused 63 ms after one of them was killed.
  - **Chat replays (Nov 16):** Qwen3-30B-A3B over all three Sparks (RDMA, direct cables) and over the three Sparks plus the desktop's CPU (RDMA and TCP in one run) are on the public page (`bench/trace/2026-11-16-chats.md`). Both were recorded while `gx10-01`'s GPU was busy with a co-tenant's job; the three-Spark one plays at 16x, the speed the same setup measured with that GPU free.
  - **Built:** `src/split.zig` (rank 0's split over any number of peers, and the expert side), `--ranks`, `--rank` and `--peer [tcp:|rdma:]IP:PORT`, per-peer capacity grants and hints, and the live diagram's N-rank layout. The split stays slow on the CPU at long prompts, since it runs one call at a time where one process batches.

---

#### [TASK-34] Vulkan GPU Backend (Intel Arc A750)
- **Milestone**: None (after v1)
- **Status**: Backlog (added Nov 17; waits for the borrowed Arc A750)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17. The GPU layer is CUDA only (`csrc/tf_gpu.c` uses CUDA's driver API, and `csrc/cuda/kernels.cu` assumes 32-lane warps), so non-NVIDIA machines join only as CPU expert nodes (TASK-22). The rest of the engine reaches the GPU through a few function pointers (`gpu_experts`, `gpu_attention`, matmul), each with a CPU fallback, so a second backend can start with matmuls and experts and add the rest later. Vulkan covers Intel, AMD and NVIDIA GPUs with one backend:
  - A backend chosen at build time (`-Dgpu=vulkan`) behind the same function pointers, with compute shaders compiled to SPIR-V and loaded at startup.
  - First the Q8_0 and f32 matmuls and the grouped expert path (TASK-12, TASK-32), then attention. Anything not yet ported falls back to the CPU and is reported, not silent (TASK-32 found Q4_K experts quietly on the CPU).
  - ggml's Vulkan shaders (llama.cpp, MIT) are the starting point for quantized formats; borrowed files keep their MIT notice and are listed in `THIRD_PARTY.md`.
  - Developed on the desktop's 5070 Ti, where Vulkan results can be compared with the CUDA backend's, then run on a borrowed Intel Arc A750 (8 GB) in a Linux machine. Native Windows builds stay out of scope, and WSL2 offers no usable Vulkan on Intel GPUs.
- **Acceptance Criteria**:
  - **Against CUDA:** on the 5070 Ti, the Vulkan backend's OLMoE-1B-7B passes the TASK-10 checks (logits within 1e-2, top-1 identical, every routing choice identical), and its matmuls agree with the CPU within 1e-3 under `TF_GPU_CHECK=1`.
  - **On the Arc A750:** OLMoE-1B-7B split between the Arc A750 and a second machine passes the TASK-10 and TASK-24 checks with the same GPU check, as TASK-26 did for Ampere, and the run is recorded in `bench/` labeled with the GPU, driver and Mesa or Intel driver version.
  - **No silent fallback:** a run prints which operations ran on the GPU, and the recorded runs show matmuls and experts there.

#### [TASK-35] Metal GPU Backend and macOS Build (Apple M4)
- **Milestone**: None (after v1)
- **Status**: Backlog (added Nov 17; waits for the borrowed M4 MacBook)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17. Apple Silicon shares memory between the CPU and GPU like the GB10, so weights can be used in place on the GPU, and Macs with large memory are natural MoE hosts. Today the engine builds only on Linux (io_uring, CUDA and RDMA) plus the Windows CPU PoC check. This task needs a macOS build first, then a Metal backend behind the same function pointers as CUDA and TASK-34:
  - **macOS build:** model loading without io_uring (plain reads or `mmap`), the shared-memory transport without `memfd` (`shm_open`), and TCP between machines. RDMA stays Linux only.
  - **Metal backend** (`-Dgpu=metal`): page-aligned weights wrapped as Metal buffers without copying, then the Q8_0 and f32 matmuls, the grouped expert path and attention, in that order. Anything not yet ported falls back to the CPU and is reported.
  - ggml's Metal shaders (llama.cpp, MIT) are the starting point for quantized formats; borrowed files keep their MIT notice and are listed in `THIRD_PARTY.md`.
  - Run on a borrowed M4 MacBook, alone and as an expert node with a Spark as rank 0 over TCP. The model is chosen by its memory: OLMoE-1B-7B fits any M4, and Qwen3-30B-A3B (Q8_0, 32 GB) fits whole only with 48 GB or more, otherwise split.
- **Acceptance Criteria**:
  - **macOS build:** `zig build` and `zig build test` pass on macOS, and OLMoE-1B-7B passes the TASK-10 checks on the CPU there, in one process and over shared memory.
  - **Metal:** OLMoE-1B-7B passes the TASK-10 and TASK-24 checks with matmuls and experts on the GPU, and its matmuls agree with the CPU within 1e-3 under `TF_GPU_CHECK=1`.
  - **Mixed cluster:** the M4 as an expert node with rank 0 on a Spark over TCP passes the same checks, and the runs are recorded in `bench/` labeled with the machine, macOS version and memory.

#### [TASK-36] Every GGUF Quant Format on the CPU and GPU
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 18)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17. The engine reads 8 weight formats (f32, f16, bf16, Q8_0, Q4_0, Q4_K, MXFP4, NVFP4), and the GPU runs only f32 and Q8_0 matmuls plus Q8_0 and Q4_K experts; everything else runs on the CPU, which TASK-32 measured at about 12 times slower. Most published GGUFs use other K-quants (a Q4_K_M file mixes Q4_K and Q6_K) or the IQ formats.
  - **CPU:** ggml's reference dequantization (llama.cpp, MIT) for the remaining formats: Q2_K, Q3_K, Q5_K, Q6_K, Q4_1, Q5_0, Q5_1, and IQ1 to IQ4 with their lookup tables, vendored as C or ported to `quant.zig`. Borrowed code keeps its MIT notice and is listed in `THIRD_PARTY.md`.
  - **GPU:** one quantized matmul and one grouped expert kernel that decode a tile of weights to f16 and multiply on the tensor cores, with a small decode function per format, in place of a kernel per format. ggml's CUDA kernels serve as the design reference. The same decode functions are reused by TASK-34 and TASK-35.
  - **No silent fallback:** a run reports any weight that stays on the CPU because its format has no GPU path (TASK-32's Q4_K experts ran on the CPU unnoticed).
- **Acceptance Criteria**:
  - **Decoding:** every format decodes test tensors exactly as ggml's reference does (bit for bit on the CPU), in `zig build test`.
  - **GPU:** every format's GPU matmul agrees with the CPU within 1e-3 under `TF_GPU_CHECK=1`.
  - **Models:** a Q4_K_M and an IQ-format GGUF of Qwen3-30B-A3B pass the TASK-10 and TASK-24 checks against transformers on the GGUF's dequantized weights, with every expert on the GPU.
- **Results (Nov 18)**: acceptance met (`bench/olmoe/2026-11-18-quant-formats.md`).
  - **Decoding:** 16 formats ported to Zig from ggml's reference decoders, keeping ggml's order of float operations; the IQ tables are generated from `ggml-common.h`; all are listed in `THIRD_PARTY.md`. A random tensor in each of 21 block formats decodes bit for bit as gguf-py does, in `zig build test` (118 tests on `gx10-02`, 100 plus 1 skipped on Windows), and ggml itself matches gguf-py bit for bit on all 21 (`tools/ref/check_ggml_quants.py`).
  - **GPU:** one decode function per format feeds one matrix-vector and one tensor-core kernel (`csrc/cuda/blocks.cuh`), for matmuls and experts, including experts whose down is in another format. `olmoe gpu-formats` checks every format but f32 against the CPU: matmuls within 1.14e-5, experts within 1.45e-4.
  - **Models:** bartowski's Q4_K_M and IQ4_XS GGUFs of Qwen3-30B-A3B pass forward and 32-token generation on the GPU, the CPU, over shared memory and through the cache, with logits within 1.85e-4. No weight ran on the CPU. The existing goldens still pass.
  - **No silent fallback:** a weight on the CPU is logged with its format, shape and reason, and counted at the end of the run.
  - **Speed:** decode at 1,024 tokens of context is 35.7 tokens/s for Q8_0, 41.9 for Q4_K_M and 48.2 for IQ4_XS. Prefill for the new formats is 20 to 30% slower than Q8_0's, whose kernel is tuned.

#### [TASK-37] SentencePiece Tokenizer and More Pre-Tokenizers
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 18)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17. The tokenizer handles byte-level BPE with 4 pre-tokenizer patterns (`gpt2`, `qwen2`, `qwen35`, `glm4`), each added by hand with a model. Models such as Mixtral use SentencePiece, and most other families need their own pattern. llama.cpp's vocabulary code (MIT) has a SentencePiece tokenizer and the patterns for dozens of model families, keyed by the GGUF's `tokenizer.ggml.pre` value, plus vocab files with expected outputs for testing.
  - SentencePiece (BPE with byte fallback, as Llama 2 era and Mixtral models use it).
  - The pre-tokenizer patterns of the families TASK-39 adds, selected from the GGUF's metadata, with a clear refusal for an unknown one.
  - Borrowed code and test vocabularies keep their MIT notice and are listed in `THIRD_PARTY.md`.
- **Acceptance Criteria**: for Mixtral, DeepSeek V3 and gpt-oss, plus every model already supported, the tokenizer matches Hugging Face on `tools/ref/check_tokenizer.py`'s strings (as TASK-31 and TASK-32 did) from the GGUF alone, and the existing models still pass.
- **Results (Nov 18)**: acceptance met (`bench/olmoe/2026-11-18-tokenizers.md`).
  - **Built** (`src/tokenizer.zig`): SentencePiece (GGUF model "llama") with byte fallback and the GGUF's prepend scheme. Its merges are rebuilt from the scores and match Mixtral's Hugging Face merges exactly, in order. Also the `deepseek-v3` and `gpt-4o` patterns, BOS from `add_bos_token`, and a refusal that names an unknown pre-tokenizer or model. Unicode tables gain letter cases, punctuation and symbols. No llama.cpp code was borrowed: Hugging Face's `tokenizers` is the reference.
  - **Checked** on `gx10-02`, 5,038 texts per model (golden prompts, 32 edge cases, 5,000 random strings, three long files), every id identical to Hugging Face's from the GGUF: Mixtral-8x7B, DeepSeek V3 and gpt-oss-20b (vocab-only GGUFs from llama.cpp's converter), OLMoE, Qwen3 (Q8_0, Q4_K_M, IQ4_XS), Qwen3.6, Qwen3.8 (vocab-only, its GGUF having been deleted) and GLM-5.3-Flash. Unit tests: 122 on `gx10-02`, 104 plus 1 skipped on Windows. Chat prompts are unchanged from the previous build.
  - **Not done:** Mixtral's chat template (TASK-38), and loading the new setups from `tokenizer.json`.

#### [TASK-38] Chat Templates Read from the GGUF
- **Milestone**: None (after v1)
- **Status**: Backlog (added Nov 17)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17. Each model's chat template is ported by hand into `src/chat.zig` (5 formats so far), so a new model cannot chat until someone writes its format. GGUFs carry the original Jinja template. llama.cpp renders them with minja, a small Jinja interpreter (MIT, C++).
  - A Jinja interpreter for the subset chat templates use (variables, loops, conditionals, filters such as `trim` and `capitalize`, string methods, `raise_exception`), as a Zig port or a vendored minja; the C++ dependency is weighed against porting.
  - The server renders the GGUF's own template, passing messages, `add_generation_prompt`, `enable_thinking` and `reasoning_effort`. A template using an unsupported feature is refused with a clear error, and the hand-written formats stay as a fallback.
  - Tool calls stay out.
- **Acceptance Criteria**: `tools/ref/check_chat.py` matches transformers' rendering for the 5 existing formats and for the templates of Mixtral, DeepSeek V3, gpt-oss, Llama 4 and Kimi K2, across its settings (system prompt or not, reasoning on and off, several turns).

#### [TASK-39] Mixtral, DeepSeek V3 and gpt-oss from llama.cpp's Specs
- **Milestone**: None (after v1)
- **Status**: Backlog (added Nov 17)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17. Each architecture is written by hand (5 so far), at 1 to 3 days each plus a transformers reference. The engine loads GGUFs made by llama.cpp's converter, so llama.cpp's per-architecture code (MIT) is the most exact record of each model's tensor names, shapes, converter transformations and forward pass; reading it replaces working out the GGUF layout by hand, as TASK-32's reference script had to. The reference is still transformers, not llama.cpp. Three popular families, each needing something new:
  - **Mixtral:** grouped-query attention with a softmax router and no shared expert; mostly a check that the existing blocks generalize, plus SentencePiece (TASK-37).
  - **DeepSeek V3 (and Kimi K2, which shares its architecture):** MLA with decoupled rope (TASK-32's MLA has none), group-limited routing, and dense leading layers.
  - **gpt-oss:** attention sinks, alternating sliding-window and full attention, and MXFP4 experts (MXFP4 already loads).
- **Acceptance Criteria**: each family passes the TASK-10 and TASK-24 checks against transformers (in float32 on the GGUF's dequantized weights, loaded lazily where the model is large), in one process and split; a tiny random model of each runs in `zig build test`; and each chats through TASK-38's templates.

---

#### [TASK-40] Native Windows Engine (olmoe)
- **Milestone**: None (after v1)
- **Status**: Backlog (added Nov 17)
- **Priority**: `could` | **Area**: `code`
- **Description**: Added Nov 17, after TASK-25 needed WSL2 plus `--advertise` and an SSH tunnel to make the desktop an expert node. Natively, Windows builds only `dispatch.exe` and `tensorfold.exe`. The forward pass already builds and passes its tests there (`olmoe.zig`, `expert_cache.zig` and the thread pool have Windows branches); what is Linux only is `olmoe_run.zig` and the modules around it.
  - **Model loading:** a Windows reader behind the same interface as the io_uring O_DIRECT one (`model_file.zig`, `csrc/io/uring_reader.c`): unbuffered `ReadFile` to load every expert, and overlapped reads on an I/O completion port for the expert cache (`--cache`).
  - **Modes:** `local` first, then `expert` and `router` over TCP (the transport already runs on Windows; the tracker import in `split.zig` becomes optional), then `shm` with `CreateProcess` and the named shared memory `dispatch.exe` already uses instead of `fork`.
  - **Server:** `serve.zig`, `telemetry.zig` and `http.zig` on Winsock (`WSAPoll`, and a loopback socket or event in place of the pipe to the engine thread), reusing `transport/tcp.zig`'s Winsock layer.
  - **GPU:** `csrc/tf_gpu.c` uses CUDA's driver API, which Windows has; `build.zig` links `nvcuda`, and nvcc needs MSVC's `cl.exe` as its host compiler.
  - **Out:** RDMA, the NVMe KV tier and saved prefixes (TASK-12, TASK-28; they could reuse the new reader later), and the dashboard and node agent (TASK-25).
- **Acceptance Criteria**:
  - **One process:** on the desktop, native `olmoe local` passes the TASK-10 and TASK-24 checks for OLMoE-1B-7B on the CPU and with `--gpu` on the 5070 Ti, and Qwen3-30B-A3B with `--cache` streaming experts from the drive.
  - **Server:** `--serve` passes `tools/ref/check_serve.py` and the built-in page shows the live diagram.
  - **Split:** a native Windows expert rank with a Spark router over TCP passes the same checks, with no WSL2 or tunnel, and `shm` passes on Windows alone.
  - **Build:** `build.ps1` builds `olmoe.exe` (with `-Gpu` for CUDA), Linux builds and tests are unchanged, and results are recorded in `bench/` with the Windows and driver versions.

#### [TASK-41] Prefetch and Cache Misses in the Live View
- **Milestone**: None (after v1)
- **Status**: Review (built Nov 18)
- **Priority**: `could` | **Area**: `webgl`
- **Description**: Added Nov 17, after the "One token" view in `web/live.html` began showing each token's trip through the layers and its experts on each machine. It cannot show prefetching yet: each step frame carries only running cache totals (`cache: {resident, hinted, late, missed, loads}`), not which expert was a hit, a prefetch or a miss, and the one cached recording (Qwen3.8-Flash-Next, TASK-31) ran without `--lookahead`, so its `hinted` count is 0 throughout.
  - **Engine, rank 0:** `ExpertCache.poll` already decides each call's status. While a viewer is connected, it writes that status into a per-step array (one byte per layer and expert, on the engine thread), and the telemetry tap, which already wraps `Moe.prefetch`, records each hint as (layer it is for, expert, layer it was sent during). Step frames gain `c: [layer, expert, status, ...]` and `h: [for layer, expert, sent during, ...]`, and the hello says which ranks have a cache. These are added fields, so the schema stays version 0 and older recordings still replay.
  - **Page:** a drive on each machine that has a cache; faint prefetch packets during a layer toward the experts predicted for a later one, over the cable for a peer's share; a load from the drive to each expert that a hint brought in; and new expert states for the key: prefetched and ready, loading from the drive, and missed (the token waits on the drive).
  - **Recording:** the TASK-31 chat again on `gx10-02` alone with `--cache 12000 --lookahead 1`, compared with the existing run without lookahead, and swapped into the published recordings (TASK-18).
  - **Optional, peers:** a peer's cache status travels back in each combine packet's reserved `flags` bits 7 to 9 (`docs/wire-format.md`), shown with a split recording that has caches on `gx10-01` or `gx10-03`, only with the owner's go-ahead and while their servers are idle.
- **Acceptance Criteria**:
  - **Feed:** unit tests cover the status array, the hint list and the frame output; with no viewer connected nothing is recorded; `docs/telemetry-schema.md` documents `c`, `h` and the hello field; the existing recordings replay unchanged.
  - **Page:** replaying the new recording, hints, drive loads, prefetched experts and misses show on the right dots, and each token's counts of resident, hinted, late and missed calls match the engine's own counters.
  - **Recording:** the new Qwen3.8 chat is published with a write-up in `bench/trace/` giving its hit rates beside the run without lookahead.
  - **Amended Nov 18** (user's decision): Qwen3.8's GGUF is no longer on `gx10-02`, which keeps GLM-5.3-Flash for TASK-32, so the recording is Qwen3-30B-A3B through a 3,000-slot cache (about half its 6,144 experts, as Qwen3.8's 12,000 of 24,576), recorded with and without `--lookahead 1`, and added to the published recordings beside the others.
- **Results (Nov 18)**: acceptance met (`bench/trace/2026-11-18-prefetch.md`); not yet deployed to the public site.
  - **Feed:** `c`, `h` and `cache_ranks` as planned (`src/telemetry.zig`, `src/expert_cache.zig`, `docs/telemetry-schema.md`). Unit tests cover the status array against the cache's counters, the hint list and the frames, and that nothing is noted without a viewer (116 tests on `gx10-02`, 98 plus 1 skipped on Windows).
  - **Page:** drives, hint packets, loads from the drive, and the prefetched, loading and missed states, with each token's counts beside the drive. Checked in Chromium in both themes and at phone width; the TASK-31 recording replays unchanged.
  - **Recording:** every step's statuses add up to the change in the engine's counters (1,182 and 1,128 steps). Lookahead cuts misses from 4.2% to 2.0% of calls, with 1.2% prefetched in time and 1.9% still loading when needed.
  - **Found and fixed:** with experts in the cache on the GPU, runs were not repeatable bit for bit (same tokens over 128 steps, logits up to 7.9e-4 apart); the two chats part at a near-tie in turn 2. Load timing split a layer into GPU jobs differently, and each job picked its kernels from its own largest expert, while the CPU added the outputs in a different order than the GPU's combine. Now each expert's GPU kernels follow its own row count, whatever job it lands in, and with a GPU the outputs are summed as the GPU does: cached runs, with or without lookahead, and a two-process split match each other and an all-in-memory GPU run bit for bit, at the same speed (`bench/olmoe/2026-11-18-cache-determinism.md`). The recordings above were made before the fix.
  - **Not done:** the optional peer part (cache status in combine packets).

---

## 4. Scope Tiers & Cut Lines

| Tier | Label | Meaning | Tasks |
|---|---|---|---|
| **Must** | `must` | The release fails without it. Together these form the **MVP: Two-Process Slice**: a real MoE runs split across two engine processes (RDMA on `gx10-03`, TCP between the desktop and `gx10-03`), loaded through `io_uring`, used from a browser chat harness, and shown in a live and recorded diagram. | 01, 02, 03, 04, 05, 06, 07, 08, 10, 15, 16, 16b, 18, 19, 23, 24, 30 |
| **Should** | `should` | The headline features beyond the MVP. Planned and expected, but droppable as a group. | 09, 11, 14, 17, 20, 21 |
| **Could** | `could` | Stretch work, attempted only when ahead of schedule or when more hardware frees up. TASK-25, 27, 28, 29, 31, 32, and 34 to 41 are post-v1. | 12, 13, 15b, 22, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41 |

**If we fall behind, we cut first:** the **Stretch** cut (`cut-1`): NVMe KV offload, compressed expert streaming, the multi-Spark run, and CPU-only expert nodes.

**Cut lines.** Cuts are applied in order. A cut task moves to `Backlog` with a `cut` note and is listed as future work in TASK-19.

| Cut | Tag | Name | Tasks dropped | Trigger |
|---|---|---|---|---|
| **First cut** | `cut-1` | **Stretch** | 12, 13, 15b, 22 (16b until Nov 13) | Alpha (`v0.2.0-alpha`) not tagged by Tue Nov 10 (not triggered: tagged Nov 10). TASK-13 alone was cut Nov 11, because measured compression ratios cannot meet its 10% bar. TASK-15b alone drops if neither `gx10-01` nor `gx10-02` is free by Sun Nov 22. |
| **Second cut** | `cut-2` | **Speculation & Swarm** | 09, 11, 14, 17 (engine runs with resident experts and a static manifest) | TASK-11 not in `Review` by Wed Nov 18 |
| **Emergency** | (none) | **TCP Floor** | Replace TASK-07 RDMA with the TASK-02 shared-memory and TCP backends for all later tasks; report those numbers as such | RDMA loopback on `gx10-03` not passing by Wed Nov 4 |

TASK-20 is `should` but untagged: it is never cut, only reduced to a minimal build guide.

**Out of scope for v1** (recorded so they are not silently added): 8-node and 1-trillion-parameter runs (only 3 Sparks exist; TASK-33 lets the engine use any number of nodes, but runs are limited to the machines available), public or unauthenticated access to the engine and the operator dashboard (TASK-25, post-v1; TASK-30's server stays on localhost or a private network), approximate expert skipping, hardware DE tuning beyond TASK-13, GPU-initiated RDMA (IBGDA), training or fine-tuning, non-NVIDIA GPU backends and macOS engine builds (post-v1: Vulkan in TASK-34 and Metal with a macOS build in TASK-35; until then non-NVIDIA machines join only as CPU expert nodes, TASK-22), and native Windows engine builds (post-v1 in TASK-40; until then `build.ps1` builds `dispatch.exe` for the CPU PoC check over shared memory and TCP, and the engine runs in WSL2).

---

## 5. Workboard Snapshot

Columns: `Backlog` | `To Do` | `In Progress` | `Review` | `Done`

| Column | Tasks |
|---|---|
| **Backlog** | TASK-13 (`cut` Nov 11; `bench/storage/2026-11-11-compression.md`), TASK-19, TASK-20, TASK-34 (Vulkan, waits for a borrowed Arc A750), TASK-35 (Metal and macOS, waits for a borrowed M4 MacBook), TASK-38 and TASK-39 (chat templates and three model families, borrowing from llama.cpp), TASK-40 (native Windows engine) |
| **To Do** | (none) |
| **In Progress** | TASK-32 (GLM-5.3-Flash, Nov 17) |
| **Review** | TASK-37 (SentencePiece and more pre-tokenizers, Nov 18; `bench/olmoe/2026-11-18-tokenizers.md`), TASK-36 (every GGUF quant format on the CPU and GPU, Nov 18; `bench/olmoe/2026-11-18-quant-formats.md`), TASK-41 (prefetch and cache misses in the live view, Nov 18; `bench/trace/2026-11-18-prefetch.md`), TASK-25 (operator dashboard, Nov 17; `bench/serve/2026-11-17-dashboard.md`), TASK-31 (Qwen3.8-Flash-Next, experts streamed from NVMe, Nov 17; `bench/olmoe/2026-11-17-qwen38.md`), TASK-29 (Qwen3.6-35B-A3B, hybrid linear attention, Nov 16; `bench/olmoe/2026-11-16-qwen35.md`), TASK-33 (any number of expert nodes, Nov 16; `bench/olmoe/2026-11-16-nodes.md`) |
| **Done** | TASK-01 to TASK-06 (PoC released as `v0.1.0-poc` on Oct 27; PoC result in `POC.md`), TASK-27 (compact KV cache formats, Nov 16; `bench/olmoe/2026-11-16-kv-formats.md`), TASK-18 (Pages showcase of recorded runs, Nov 16; https://aiforgames2026.github.io/DistributedMoEEngine/), TASK-17 (capacity and memory guardrails, Nov 16; `bench/rc/2026-11-16-guardrails.md`), TASK-15 (one-hour soak and cross-machine run, Nov 16; `bench/rc/2026-11-16-soak.md`), TASK-16 (recording and replay, Nov 15; `bench/trace/2026-11-15-recording.md`), TASK-30 (server and built-in chat harness, Nov 15; `bench/serve/2026-11-15-server-latency.md`), TASK-16b (live telemetry, Nov 15; `bench/serve/2026-11-15-live-telemetry.md`), TASK-23 (Testing Session 1, Nov 13; `bench/playtest/session-1.md`), TASK-07 (acceptance met Nov 5; `bench/transport/2026-11-05.md`), TASK-08 (acceptance met Nov 5; `bench/storage/gx10-03-2026-11-05.md`), TASK-10 (acceptance met Nov 5; `bench/olmoe/2026-11-05.md`), TASK-21 (direct TCP path, Nov 5; `bench/transport/2026-11-05-direct-path.md`), TASK-26 (Ampere run passed Nov 10; `bench/olmoe/2026-11-10-ampere.md`), TASK-24 (generation over RDMA passed Nov 10; `bench/olmoe/2026-11-10-generate.md`), TASK-11 (lookahead and expert cache passed on `gx10-03` Nov 10; `bench/olmoe/2026-11-10-lookahead.md`), TASK-14 (30-minute forced-misprediction soak passed on `gx10-03` Nov 11; `bench/olmoe/2026-11-10-miss-handling.md`), TASK-15b (two Sparks over RDMA, Nov 11; `bench/rc/2026-11-11-two-sparks.md`; one unexplained differing round in an earlier repeat), TASK-09 (tracker across two Sparks, Nov 11; `bench/tracker/2026-11-11.md`), TASK-22 (CPU expert on the desktop with a GB10 router, Nov 11; `bench/olmoe/2026-11-11-cpu-expert.md`), TASK-12 (128k context with the KV middle on NVMe, Qwen3-30B-A3B, Nov 11; `bench/olmoe/2026-11-11-long-context.md`), TASK-28 (saved prompt prefixes, 128k first token in 4.2 s instead of 510 s, Nov 11; `bench/olmoe/2026-11-11-prefix-cache.md`) |

| Area | Tasks | Count |
|---|---|---|
| `code` | 01, 02, 03, 07, 08, 10, 11, 12, 13, 14, 17, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 | 28 |
| `architecture` | 06, 09 | 2 |
| `benchmarking` | 05, 15, 15b, 21 | 4 |
| `webgl` | 04, 16, 16b, 23, 25, 41 | 6 |
| `docs` | 18, 19, 20 | 3 |

| Priority | Count | Cut tag |
|---|---|---|
| `must` | 17 | none |
| `should` | 6 | 4 tagged `cut-2` |
| `could` | 20 | 4 tagged `cut-1`; TASK-25, 27, 28, 29, 31, 32, and 33 to 41 are post-v1; TASK-26 untagged |

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
| TASK-15 | 10, 24, time agreed with co-tenant | 16, 19 |
| TASK-15b | 15, a free and cabled second Spark | 19 |
| TASK-16 | 04, 15, 30 | 18 |
| TASK-16b | 16 (schema), 30 | 25 |
| TASK-17 | 10 | 15 (hardening) |
| TASK-18 | 16 | Final |
| TASK-19 | 15 | Final |
| TASK-20 | all code tasks | Final |
| TASK-22 | 08, 10 | 15 (optional CPU node in the cross-machine run) |
| TASK-23 | 04, 21 (the direct link it teaches) | Testing Session 1 |
| TASK-24 | 10 | 15 |
| TASK-25 | 09, 16b, 30, v1 release | (none) |
| TASK-26 | 10 | 09 (classroom machines as swarm nodes) |
| TASK-27 | 24 | (none) |
| TASK-28 | 03, 24 | (none) |
| TASK-29 | 12 (Qwen3-MoE support) | 31 |
| TASK-30 | 09 (split runs), 24, 28 | 16, 16b, 18 |
| TASK-31 | 29 (Gated DeltaNet), 11 and 14 (expert cache), 33 (more than two nodes) | 32 |
| TASK-32 | 31 (delta-rule kernels, sparse attention) | (none) |
| TASK-33 | 09 (tracker), 10, 16b (live diagram), 17 (capacity) | 31 |

**Critical path (MVP)**: TASK-05 → TASK-01 → TASK-02 → TASK-07 → TASK-10 → TASK-24 → TASK-30 → TASK-16 → TASK-18 (TASK-15 feeds TASK-16 in parallel).

---

## 7. Risks, Research-Verified Constraints & Fallbacks

### 7.1 Top 3 Risks (each tested in the PoC)

| # | Risk | PoC test (task) | Trigger date & condition | Fallback |
|---|---|---|---|---|
| **R1** | **Zig is new to the only engineer.** Zig is pre-1.0, with breaking changes between versions and fewer learning resources; combining it with C, CUDA, liburing, and verbs could stall the schedule. | TASK-01 toolchain gate: `build.zig` compiles C, links `liburing` and `libcudart`, calls CUDA on the 5070 Ti, cross-compiles to aarch64, and runs on `gx10-03`. | **Wed Oct 28**: gate not passing, or the TASK-01 matmul not working by Fri Oct 30. | Write the engine core in C11 with CMake (reusing the existing CLion project). No task changes, because every subsystem already sits behind a C ABI; Zig can return later for the scheduler only. |
| **R2** | **Only one node, and it is shared.** **Partly retired Nov 11:** `gx10-02` freed up and the two-node RDMA run passed (TASK-15b). `gx10-01` and `gx10-02` are busy, so real node-to-node RDMA may never be available; `gx10-03`'s ConnectX-7 ports may be unlinked, which can block even RDMA loopback; the other student's jobs compete for memory, GPU, and disk. | TASK-05 probe: ConnectX-7 port state, loopback `ib_write_lat` on `gx10-03`, and realistic free memory; TASK-02 TCP backend between the desktop and `gx10-03`. | **Fri Oct 30**: loopback `ib_write_lat` does not run on `gx10-03`, or less than about 40 GB of memory is realistically free. **Memory condition already met on Oct 26 (about 11 GB free).** | RDMA code is developed against Soft-RoCE (`rdma_rxe`) and performance runs use the shared-memory and TCP backends if loopback fails (Emergency cut); for memory, cap each engine process at 4 GB on `gx10-03`, develop and test models on the desktop (64 GB RAM), use OLMoE-1B-7B-class models on the Spark, and ask the co-tenant for a window with vLLM paused for the soak; label every benchmark with the hardware and co-tenant load it ran under. |
| **R3** | **NVMe expert streaming is too slow or unsupported.** Direct I/O alignment may not be supported by `gx10-03`'s filesystem, or throughput may fall far below what streaming needs. **Retired Oct 27 for single-drive reads:** ext4 reports DIO alignment and the reader reaches 7.2 GB/s, matching `fio`; contention with GPU kernels is still untested. | TASK-03 reader versus the `fio` O_DIRECT baseline on `gx10-03`. | **Fri Oct 30**: O_DIRECT reads fail, or the reader reaches under 50% of the `fio` baseline. | Load every expert at startup and run resident-only (the RC model fits in memory); TASK-12 and TASK-13 drop with the first cut, and TASK-11 prefetching targets the transport only. |

### 7.2 Other Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R4 | One engineer for 22 tasks in 6 weeks | High | High | Cut lines in Section 4 with dated triggers; AI pair for boilerplate, tests, and review |
| R5 | Co-tenant load makes `gx10-03` benchmarks noisy | High | Medium | Record co-tenant GPU and memory use with each run; repeat runs and report the median; agree on a quiet window for the soak |
| R6 | Lookahead recall below 80% at H = 2 | Medium | Medium | Report H = 1; exact-mode miss handling keeps output correct |
| R7 | Kernel code written on the 5070 Ti (12.0) behaves differently on GB10 (12.1) | Medium | Medium | Build for both `sm_120` and `sm_121`; run correctness tests on both; never publish desktop numbers as GB10 numbers |
| R8 | GPLv3 vs. proprietary NVIDIA libraries (CUDA runtime, nvCOMP) in distributed binaries | Medium | Medium | Add a GPLv3 section 7 linking exception for NVIDIA libraries before the first release. Lumabri (Apache-2.0) is not vendored: TASK-09's tracker is written in Zig and follows its design |
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