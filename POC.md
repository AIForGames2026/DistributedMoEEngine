# Proof of Concept: TensorFold-Swarm

**Result: PASSED**, with one risk trigger tripped (free memory on `gx10-03`) and its planned fallback now in effect. Evidence is below; every number comes from a recorded run in [`bench/`](bench/).

## 1. The assumption under test

> **We believe a Mixture-of-Experts model can be split across separate engine processes, with each token's hidden state dispatched to the process that owns its experts and combined back over our own slot-ring transport, giving the same output as a single-process run on the hardware we actually have (one shared DGX Spark, `gx10-03`, plus the RTX 5070 Ti desktop). If it doesn't, the project should pivot to a single-process, single-node MoE engine (experts resident in memory or streamed from NVMe) with the WebGL inspector replaying its routing.**

### Why this is the riskiest assumption

Everything that makes this project more than "run a model on one GPU" depends on it. It is the union of the three top risks in [Plan.md section 7.1](Plan.md#71-top-3-risks-each-tested-in-the-poc), and each one alone could have sunk the distributed design:

| Risk | What could go wrong | How the PoC tests it |
|---|---|---|
| **R2: one shared node** | `gx10-01` and `gx10-02` are busy, so there may be no second machine to split a model across, the ConnectX-7 ports may not even loop back, and the co-tenant's vLLM competes for memory. | Split the model across two processes on `gx10-03` and across the desktop and `gx10-03` over TCP (TASK-02, TASK-10 slice); probe the ports and RDMA loopback (TASK-05). |
| **R1: Zig is new to the only engineer** | The engine core is Zig 0.17 (pre-1.0) mixed with C, CUDA, liburing, and verbs. If that toolchain does not hold together, nothing else gets built. | Toolchain gate on both machines (TASK-01). |
| **R3: NVMe expert streaming too slow** | If `O_DIRECT` reads fail or run far below the drive, experts cannot be streamed and every model must fit in memory. | io_uring reader versus `fio` on `gx10-03` (TASK-03). |

The fallback for the whole assumption is the pivot in the sentence above. Each risk also has its own narrower fallback in Plan.md (C11 instead of Zig, Soft-RoCE plus a 4 GB memory cap, resident-only experts).

### Pass/fail criteria (decided before building)

These were committed in Plan.md on **Oct 26, 2026 at 12:35** ([`0cf12dd`](https://github.com/AIForGames2026/DistributedMoEEngine/blob/0cf12dd/Plan.md)), about 12 hours before the first line of engine code ([`0afbf93`](https://github.com/AIForGames2026/DistributedMoEEngine/commit/0afbf93), Oct 27 00:29).

The core check: **a forward pass of the synthetic MoE split across two engine processes produces output logits within tolerance of the PyTorch reference, with identical top-1 next tokens** (TASK-10's criterion; target max abs error at or below 1e-2). The router process may only learn what the expert process computed through packets on the wire.

Supporting checks, with the trigger dates that would have forced a cut:

| Check | Pass | Fail trigger (from the risk table) |
|---|---|---|
| Transport (TASK-02) | 4,096 dispatch + 4,096 combine packets, every one passing CRC and generation checks, over shared memory and over TCP desktop to `gx10-03` | (R2) |
| Toolchain gate (TASK-01) | `zig build` links liburing and CUDA, calls both, cross-compiles to aarch64, runs on `gx10-03` | Not passing by Wed Oct 28 |
| RDMA loopback (TASK-05) | `ib_write_lat` runs on `gx10-03` | Does not run by Fri Oct 30 |
| Free memory (TASK-05) | About 40 GB realistically free on `gx10-03` | Less than about 40 GB |
| NVMe reader (TASK-03) | `O_DIRECT` works and reaches at least 50% of `fio` | Fails, or under 50% of `fio` by Fri Oct 30 |

## 2. Running the PoC

Linux or WSL2 with Zig 0.17.0 (`bash tools/setup/install-zig.sh` installs it without sudo). The core check needs nothing else:

```bash
zig build install -Doptimize=ReleaseFast
zig-out/bin/dispatch shm --moe
```

Across two machines (start the expert first; port 7470 must be reachable):

```bash
zig-out/bin/dispatch expert --port 7470 --moe            # on gx10-03
zig-out/bin/dispatch router --host <expert IPv4> --port 7470 --moe   # on the desktop
```

Prebuilt Linux binaries (x86_64 and aarch64) are on [GitHub Releases](https://github.com/AIForGames2026/DistributedMoEEngine/releases). The other checks (`zig build gate`, `io-test`, `io-bench`, `tools/probe/probe.py`) are in the [README](README.md#running-it).

**What `--moe` does:** both processes embed the tiny MoE from [`tests/golden/moe_tiny.tfref`](tests/golden/) (2 layers, 8 experts, top-2, 16 tokens), written by the seeded PyTorch harness in [`tools/ref/ref_moe.py`](tools/ref/ref_moe.py). Rank 0 owns the norms, the router, and the LM head. Rank 1 owns every expert's FFN. For each layer, rank 0 routes every token, sends one 7,552-byte [wire-format](docs/wire-format.md) packet per (token, chosen expert), rank 1 runs the expert and sends the result back, and rank 0 folds the weighted results into the hidden state. At the end rank 0 compares routing, hidden states, and logits against the PyTorch goldens and prints PASS or exits with an error. Code: [`src/dispatch.zig`](src/dispatch.zig) (`moeRouter`, `moeExpert`), [`src/moe_ref.zig`](src/moe_ref.zig) (`route`, `expert`).

## 3. Demo video

[Demo video (2 min)](VIDEO_LINK_HERE): the two-process MoE passing on shared memory and desktop to Spark, the inspector replaying routing at 144 FPS, and the platform probe answers.

## 4. Playable build

- **Browser:** the WebGL inspector at https://aiforgames2026.github.io/DistributedMoEEngine/ plays telemetry traces of the same dispatch/compute/combine flow, with timing from the measured numbers below. Nothing to install.
- **Engine:** [GitHub Releases](https://github.com/AIForGames2026/DistributedMoEEngine/releases) has Linux ELF binaries for x86_64 (desktop, WSL2) and aarch64 (DGX Spark) with SHA-256 sums. No Windows `.exe`.

**Why the engine can't run in a browser:** it is the thing being tested, and it is made of what a browser sandbox forbids: raw sockets between machines, `fork` and shared memory between processes, `io_uring` with `O_DIRECT` on an NVMe drive, and (from Alpha) CUDA and RDMA verbs on the ConnectX-7. A WebAssembly build would have none of those, so it would test nothing. The inspector is the web part of the project by design (Plan.md section 1: visitors never download anything).

## 5. Result

**Passed.** The core check and every supporting check passed; one trigger (free memory) tripped on Oct 26 and its fallback is in effect.

### Core check: two-process MoE (Nov 2)

| Path | Routing (64 expert calls) | Hidden state max err | Logits max err | Top-1 next token | Time | Both exit 0 |
|---|---|---|---|---|---|---|
| Shared memory, WSL2 desktop | identical | 4.8e-7 | 7.2e-7 | 16/16 | 0.63 ms | yes |
| Shared memory, `gx10-03` | identical | 4.8e-7 | 7.2e-7 | 16/16 | 0.93 ms | yes |
| TCP loopback, WSL2 desktop | identical | 4.8e-7 | 7.2e-7 | 16/16 | 1.31 ms | yes |
| TCP, desktop to `gx10-03` (Tailscale relay) | identical | 4.8e-7 | 7.2e-7 | 16/16 | 556 ms | yes, 4 of 4 runs after the close fix |

The target was 1e-2; the result is 7.2e-7, which is float32 summation-order noise. The values are identical on every path because the packets carry exact float32 bits. Full log: [`bench/transport/2026-11-02.md`](bench/transport/2026-11-02.md).

### Supporting checks (Oct 26 to 27)

| Check | Result | Evidence |
|---|---|---|
| Transport | 4,096 + 4,096 packets verified on all three paths; 1.8 µs per token over shared memory on `gx10-03` | [`bench/transport/2026-10-27.md`](bench/transport/2026-10-27.md) |
| Toolchain gate | Passed Oct 27 (a day early) in WSL2 and on `gx10-03`; start to first kernel 0.30 ms. **R1 retired.** | [Plan.md TASK-01](Plan.md) |
| RDMA loopback | Runs: 3.17 µs for a 7,552-byte packet, 4 ports active | [`bench/platform/gx10-03-2026-10-27.json`](bench/platform/gx10-03-2026-10-27.json) |
| Free memory | **Tripped:** about 11 to 12 GB free (co-tenant vLLM resident) | [`bench/platform/gx10-03-2026-10-26.md`](bench/platform/gx10-03-2026-10-26.md) |
| NVMe reader | 7.24 to 7.27 GB/s, 100% of `fio`. **R3 retired** for single-drive reads. | [`bench/storage/gx10-03-2026-10-27.md`](bench/storage/gx10-03-2026-10-27.md) |
| Reference harness | Seeded, reproducible; the Zig CPU MoE matches it layer by layer | `zig build test`, `tools/ref/ref_moe.py --check` |

### What this does not prove yet

- Our transport has not run over RDMA. RDMA loopback was measured with `perftest`; the verbs backend is TASK-07 (Alpha).
- The model is a 2-layer synthetic MoE on CPU. A real model (OLMoE-1B-7B) and GPU kernels are TASK-08 and TASK-10.
- The desktop-to-Spark numbers measure the Tailscale relay (33 ms round trip), not the transport.
- The inspector's traces are synthetic; the engine starts recording real ones in TASK-16.

### What the PoC revealed that we didn't expect

1. **GB10 is not a B200.** No Tensor Memory (`tcgen05`), no hardware decompression engine (mask 0), no GPUDirect RDMA. FP4 kernels use warp-level MMA with 99 KiB of shared memory; compressed streaming (TASK-13) is software-only; the NIC registers host memory (the same LPDDR5X the GPU reads).
2. **Only about 12 GB of the Spark's 128 GB is free** because of the co-tenant. The R2 fallback applies: OLMoE-1B-7B-class models, a 4 GB cap per engine process, and model development on the desktop.
3. **Integrity checking was the bottleneck, not the transport.** Software CRC32C cost 30 µs per token on ARM; the ARMv8 CRC instructions brought it to 1.8 µs. A generic aarch64 cross-compile silently loses those instructions (44 µs per token), so release builds now target `-Dcpu=cortex_a78`.
4. **The relayed desktop-to-Spark path drops the last bytes before a TCP FIN.** It first lost the final combines (fixed Oct 27 with a goodbye handshake), then, found while preparing this PoC, the router's own goodbye. The handshake is now ordered (rank 0 first, rank 1 closes after rank 0), and CI checks both processes' exit codes. RDMA teardown in TASK-07 will need the same ordering.
5. **There is no direct network path** from the desktop to the Sparks (`tailscale ping` never goes direct), so cross-machine latency is a property of the relay until that changes.

## 6. Scope update

The assumption passed, so the Must/Should/Could tiers in [Plan.md section 4](Plan.md#4-scope-tiers--cut-lines) stand unchanged. The tripped memory trigger changes model size, not scope; it was already the planned fallback.

Alpha (Nov 8) board changes from this result:

- **TASK-10** starts ahead of schedule: the synthetic-MoE, two-process, CPU slice passes over shared memory and TCP. What's left: GPU kernels, the RDMA path, and OLMoE-1B-7B.
- **TASK-07** carries two new requirements: the ordered goodbye, and CRC on the hardware path (or offloaded).
- **TASK-21 (new, `should`): direct desktop-to-Spark path.** Get a direct Tailscale connection (or another route) and re-measure, so cross-machine numbers stop measuring the relay.
