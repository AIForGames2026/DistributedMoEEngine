# Native Windows engine, first step (2026-11-20)

TASK-40. `olmoe.exe local` built and run on Windows itself, without WSL2. This step covers model loading and the one-process mode on the CPU. The GPU, the server, and the split modes come next.

## Machine

- Windows 11 Home 10.0.26200, Intel Core i7-12700K (20 threads), Zig 0.17.0, target `x86_64-windows-gnu`.
- Model: OLMoE-1B-7B-0924-Instruct Q8_0 (7.36 GB), copied from WSL2 to `E:\models\olmoe`, a 16 TB Toshiba MG08 hard drive.
- Built with `-Doptimize=ReleaseFast -Dcpu=native`; `build.ps1` builds `x86_64_v2` and passes the same TASK-10 check.

## What was built

- **Reader:** `csrc/io/win_reader.c` implements `tf_io.h` with `FILE_FLAG_NO_BUFFERING` and up to 16 overlapped `ReadFile` calls in flight. The alignment is the volume's sector size, at least 4096. `build.zig` uses it in place of the io_uring reader on Windows, for `olmoe` and `tfload`.
- **Files and process calls:** `model_file.zig` opens and reads files with Win32. `src/plat.zig` holds what `olmoe_run.zig` called through libc: environment variables, the clock, the host name, file writes, and resident memory.
- **Threads:** the CPU thread pool (`pool.zig`) only had pthreads, so on Windows every matmul ran on one thread. It now uses SRW locks and condition variables there.
- **Not on Windows yet:** `--serve` and `--record`, the `expert`, `router`, `shm` and `rdma` modes, `dashboard` and `node`, `--kv-offload` and `--prefix-cache`. Each stops with a message.

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
