# Native Windows engine: one process, split modes and GPU (2026-11-20)

TASK-40. `olmoe.exe` built and run on Windows itself, without WSL2. The first step covers model loading and the one-process mode on the CPU, the second the split modes (`shm`, and `expert` and `router` over TCP), and the third the GPU. The server comes next.

## Machine

- Windows 11 Home 10.0.26200, Intel Core i7-12700K (20 threads), Zig 0.17.0, target `x86_64-windows-gnu`.
- Model: OLMoE-1B-7B-0924-Instruct Q8_0 (7.36 GB), copied from WSL2 to `E:\models\olmoe`, a 16 TB Toshiba MG08 hard drive.
- Built with `-Doptimize=ReleaseFast -Dcpu=native`; `build.ps1` builds `x86_64_v2` and passes the same TASK-10 check.

## What was built

- **Reader:** `csrc/io/win_reader.c` implements `tf_io.h` with `FILE_FLAG_NO_BUFFERING` and up to 16 overlapped `ReadFile` calls in flight. The alignment is the volume's sector size, at least 4096. `build.zig` uses it in place of the io_uring reader on Windows, for `olmoe` and `tfload`.
- **Files and process calls:** `model_file.zig` opens and reads files with Win32. `src/plat.zig` holds what `olmoe_run.zig` called through libc: environment variables, the clock, the host name, file writes, and resident memory.
- **Threads:** the CPU thread pool (`pool.zig`) only had pthreads, so on Windows every matmul ran on one thread. It now uses SRW locks and condition variables there.
- **Not on Windows yet:** `--serve` and `--record`, `rdma` and the tracker, `dashboard` and `node`, `--kv-offload` and `--prefix-cache`. Each stops with a message.

## Results

| Check | Golden | Result |
|---|---|---|
| TASK-10 forward pass | `olmoe_forward.tfref` | PASS: logits within 3.77e-5, every top-1 token and expert choice identical |
| TASK-24 generation | `olmoe_generate.tfref --generate 64` | PASS: 3 x 64 tokens match `generate()`, logits within 4.77e-5 |
| TASK-24 through the expert cache | the same, `--cache 512` | PASS: the same tokens and logits |
| Unit tests | `zig build test` | 116 pass, 1 skipped |
| `build.ps1 -Model ... -Golden olmoe_forward.tfref` | | every step passes |

The errors match the Linux run's to every printed digit.

| Run on this desktop | Load | Generation |
|---|---|---|
| Native, one thread (before the pool fix) | 28.5 s | 2.7 tokens/s |
| Native, 20 threads | 28.5 s | 14.0 tokens/s |
| WSL2, the same source (`zig build olmoe -Doptimize=ReleaseFast`) | 1.1 s | 5.9 tokens/s |

- **Load:** the native load reads the hard drive at 258 MB/s. The WSL2 load reads its own copy on the virtual disk (NVMe), which the page cache probably already held.
- **Generation:** WSL2 is 2.4 times slower on the same CPU. This was not looked into.
- **Expert cache:** 0.5 to 0.7 tokens/s. 38% of calls missed, and each load from the hard drive took 90 ms. The Qwen3-30B-A3B check with `--cache` should run from one of the NVMe drives.
- **Older golden:** `olmoe_golden.tfref` (Nov 5) stops with `NotFound` on Windows and Linux. The documented check uses `olmoe_forward.tfref`.

## Linux unchanged

In WSL2 (Ubuntu 24.04), `zig build test` passes and `olmoe local` passes TASK-10 and TASK-24 with the same numbers. The RDMA and GPU builds on `gx10-02` were not rebuilt for this step. In the shared files (`pool.zig`, `olmoe.zig`, `model_file.zig`, `olmoe_run.zig`), the Linux code paths are the same, except for moving the libc calls into `plat.zig` and the pool's into small lock wrappers.

## Split modes (second step)

- **shm:** Windows has no `fork`. Rank 0 makes a named shared region for each expert rank, as `dispatch.exe` already did, and starts it as `olmoe shm-expert MODEL ... --shm-name NAME --rank R`, with the options rank 0 was given. If rank 0 fails, it kills the expert ranks, which would otherwise wait for packets forever.
- **TCP:** each Winsock call costs much more than a Linux one, and the link made a poll, two receives and two sends per packet. On Windows it now sends the frame header and packet in one call, and receives through a 256 KB buffer. Linux keeps its code (`MSG_MORE`, direct reads).
- **Memory reports:** `budget.residentBytes` reads the working set on Windows, so expert ranks report resident memory there too (before, 0 MB).

All runs use `olmoe_generate.tfref --generate 64 --payload f32`. fp8 and bf16 payloads miss the 1e-2 logits target on every platform, a known issue since the Beta.

| Run | Result | Generation |
|---|---|---|
| `shm`, 2 ranks | PASS: 64/64 tokens on all 3 prompts, logits within 4.77e-5 | 13.1 to 15.6 tokens/s |
| `shm`, 3 ranks | PASS | 10.6 to 15.6 tokens/s |
| TCP loopback, before the buffered link | PASS | 3.1 to 3.8 tokens/s |
| TCP loopback, buffered | PASS | 11.8 to 14.3 tokens/s |
| WSL2 TCP loopback, for comparison | PASS | 6.2 to 7.6 tokens/s |
| `gx10-02` router, native Windows expert over Tailscale | PASS: 64/64 tokens on all 3 prompts, logits within 4.77e-5 | 0.5 tokens/s |

- **The Spark run:** it needed no WSL2, `--advertise` or SSH tunnel. Its speed is the link's: `tailscale ping` showed no direct path, only Tailscale's New York relay, at 31 to 33 ms per round trip. TASK-22's runs had a direct path under 1 ms one way. The expert spent 1.08 ms per call (0.52 ms on loopback), 18.5 s in all of the run's roughly 6.5 minutes.
- **Linux:** 134 tests pass in WSL2 and on `gx10-02` (built with `-Drdma=true -Dgpu=true`), and the forked `shm` with 3 ranks still passes in WSL2.
- **`build.ps1 -Model ... -Golden ...`** now also runs `olmoe shm` with f32 payloads. Every step passes.

## GPU (third step)

- **Toolchain:** CUDA 13.1 (V13.1.115), NVIDIA driver 617.42, Visual Studio 2022's MSVC 14.44 as nvcc's host compiler. CUDA 13.1 refuses MSVC 19.50 and newer, so Visual Studio 2026's 14.51 cannot be used; `build.ps1 -Gpu` picks the newest 14.4x.
- **Linking:** the toolkit's `cuda.lib` carries `/DEFAULTLIB` entries for `LIBCMT`, `OLDNAMES` and `uuid`, MSVC's runtime, which a `x86_64-windows-gnu` link does not have. `build.zig` instead makes an import library for `nvcuda.dll` with `zig dlltool` from `csrc/cuda/nvcuda.def`, which lists the 25 driver functions `tf_gpu.c` calls. A new driver call needs a line there.
- **Code:** `tf_gpu.c` built for Windows unchanged; `gpu.zig` reads its environment variables through `plat.zig`.

| Check | Result |
|---|---|
| OLMoE TASK-10, `--gpu` | PASS: logits within 4.67e-5, every expert choice identical |
| OLMoE TASK-24, `--gpu --generate 64` | PASS: 64/64 tokens on all 3 prompts, 110 to 123 tokens/s (CPU: 14) |
| OLMoE `shm --gpu --payload f32 --generate 64` | PASS, 70 to 75 tokens/s |
| `olmoe gpu-formats` | PASS: every weight format's GPU kernels agree with the CPU within 1e-3 |
| Qwen3-30B-A3B Q8_0, `--cache 3000 --gpu`, from the NVMe drive | PASS: logits within 2.67e-4, every expert choice identical, 26 s |
| The same without `--gpu` | PASS: logits within 2.65e-4, 25 s |
| `build.ps1 -Gpu -Model ... -Golden olmoe_forward.tfref` | every step passes |

- **Qwen3-30B-A3B:** 32.5 GB, copied from WSL2 to `D:\models\qwen3-30b-a3b` (WD_BLACK SN7100). 3,000 slots (15.1 GB) hold about half of its 6,144 experts. The run loaded 20.6 GB at about 3 ms per expert (5 MB each); OLMoE's 6.7 MB experts took 90 ms each from the hard drive. As on Linux, a discrete GPU leaves cached experts on the CPU (only GB10's shared memory maps them), and the run reports the unmapped calls.
- **Linux:** on `gx10-02`, `-Drdma=true -Dgpu=true` builds `olmoe` and `attbench`, every test step succeeds, and OLMoE's GPU generate check passes (102 to 105 tokens/s). WSL2's CPU build and tests pass.
