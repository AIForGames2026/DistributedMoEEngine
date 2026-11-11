# Building and running everything

The [README](../README.md) has the quick version. This page covers every program, check, and benchmark.

## Web inspector

Requires Node.js 22 or newer.

```bash
cd web
npm ci
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173). `npm run build` writes the static site to `web/dist/`. `npm test` checks the trace generator and that the guided mission's lessons still hold.

## Engine

Requires Zig 0.17.0 (pinned in `.zig-version`) on Linux or WSL2. One-time setup on WSL2 (Ubuntu 24.04), from the repo root:

```bash
bash tools/setup/install-zig.sh
```

```bash
sudo bash tools/setup/wsl2-packages.sh
```

The first installs Zig into `~/tools/zig` (checksum verified, no sudo). The second installs liburing, rdma-core, fio, and the CUDA 13.0 toolkit from NVIDIA's WSL repository (the GPU driver comes from Windows).

```bash
zig build test --summary all
```

Toolchain gate (links liburing and the CUDA runtime, then calls both). Add `-During=<prefix>` if liburing is not in the system paths, as on the Sparks:

```bash
zig build gate
```

io_uring expert reader (TASK-03): unit test, then a throughput run against a 1 GiB pattern file (delete it afterward):

```bash
zig build io-test
```

```bash
zig build io-bench -Doptimize=ReleaseFast && zig-out/bin/uring_bench mk /tmp/expert.bin 1024 && zig-out/bin/uring_bench read /tmp/expert.bin --bs 256 --qd 16 --verify
```

Reference harness (TASK-06, testing only): regenerates `tests/golden/moe_tiny.tfref` from a fixed seed, or checks that the committed file still matches. Needs PyTorch (CPU build), for example in a virtual environment:

```bash
python3 -m venv ~/tools/ref-venv && ~/tools/ref-venv/bin/pip install -r tools/ref/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
~/tools/ref-venv/bin/python tools/ref/ref_moe.py --check
```

`tools/ref/ref_quant.py --check` does the same for the expert loader's fixtures (TASK-08): a GGUF file with Q8_0, MXFP4, and NVFP4 expert tensors written with llama.cpp's gguf-py, and a ModelOpt NVFP4 safetensors checkpoint.

Expert loader (TASK-08): `tfload` reads only the experts a process owns from a GGUF or ModelOpt NVFP4 safetensors checkpoint, through the io_uring reader, and checks each one against a CRC32C index written once by `tfload index`. For example, with allenai's [OLMoE-1B-7B-0924-Instruct Q8_0 GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF) (7.4 GB) in `~/models`:

```bash
zig build tfload -Doptimize=ReleaseFast && zig-out/bin/tfload index ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf && zig-out/bin/tfload load ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf --experts 0-31
```

Add `--golden FILE.tfref` (from `ref_quant.py --model ... --out FILE.tfref`) to compare the dequantized experts with llama.cpp's own dequantizer. Results: [`bench/storage/gx10-03-2026-11-05.md`](../bench/storage/gx10-03-2026-11-05.md).

OLMoE-1B-7B forward pass (TASK-10): `olmoe` runs the model split across two processes, rank 0 with attention, router, LM head, and experts 0 to 31, rank 1 with experts 32 to 63, and compares the output with Hugging Face transformers on the same weights. Make the reference once (needs the model's `config.json` and `tokenizer.json` from [allenai/OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)):

```bash
~/tools/ref-venv/bin/python tools/ref/ref_olmoe.py --gguf ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf --config ~/models/config.json --tokenizer ~/models/tokenizer.json --out ~/models/olmoe_forward.tfref
```

Then, for example over RDMA with CUDA matmuls (drop `-Dgpu=true` and `--gpu` for CPU only; `shm`, `local`, and `expert` / `router` over TCP work the same way):

```bash
zig build olmoe -Doptimize=ReleaseFast -Drdma=true -Dgpu=true && zig-out/bin/olmoe rdma ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf --golden ~/models/olmoe_forward.tfref --payload f32 --gpu
```

Qwen3-MoE and long context (TASK-12): every `olmoe` mode also runs Qwen3-30B-A3B from Qwen's [Q8_0 GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) (32.5 GB). It has grouped KV heads, per-head QK norm, and renormalized top-8 routing. `--yarn 4` turns on YaRN rope scaling, which Qwen documents for 128k tokens (4 × its 32,768 native positions). `tools/ref/ref_qwen3.py` makes the goldens, with transformers' own `Qwen3MoeForCausalLM`; it takes the same arguments as `ref_olmoe.py`, plus `--yarn F`. With `--gpu`, attention runs on the GPU too, reading the KV cache in place, and each layer's experts run in one GPU call. `TF_GPU_ATTENTION=0` keeps attention on the CPU, and `TF_ATT_KERNEL=1` selects the first, simpler attention kernel. Large CPU matmuls (the f32 router, say) use every CPU thread; `TF_THREADS=N` sets how many. `zig build attbench -Dgpu=true` builds a benchmark of both attention kernels (`attbench --pos 16384,65536`).

For a long prompt, `tools/ref/make_tokens.py --tokenizer tokenizer.json --out long.tokens FILE...` tokenizes text files. Then `--context N --tokens long.tokens` prefills N tokens `--chunk` at a time (default 2048), decodes `--generate M` tokens greedily, and reports the time to first token and the decode speed. `--kv-offload DIR` keeps only the first 128 positions and the most recent `--kv-window` (default 8192) in memory and moves the rest of the KV cache to a file in DIR on NVMe. Each layer's middle is read back through the io_uring reader while the previous layer runs. `--save F` and `--compare F` check that two runs, say all in memory and offloaded, produce the same tokens and logits bit for bit:

```bash
zig-out/bin/olmoe local Qwen3-30B-A3B-Q8_0.gguf --gpu --yarn 4 --tokens long.tokens --context 131040 --generate 32 --save ram.bin
```

```bash
zig-out/bin/olmoe local Qwen3-30B-A3B-Q8_0.gguf --gpu --yarn 4 --tokens long.tokens --context 131040 --generate 32 --kv-offload /path/on/nvme --compare ram.bin
```

CPU-only expert node (TASK-22): any `olmoe` process run without `--gpu` (or built without `-Dgpu=true`) runs its experts on the CPU, so a machine with no NVIDIA GPU can be rank 1, for example `olmoe expert` in WSL2 on a desktop with a GPU router on a Spark. When the Spark has to dial in through a relay on Windows, point the relay at the WSL2 VM's own address (`hostname -I` in WSL2), not `localhost`: WSL2's localhost relay made generation 3.7x slower ([`bench/olmoe/2026-11-11-cpu-expert.md`](../bench/olmoe/2026-11-11-cpu-expert.md)).

`--payload f32` is exact (two packets per expert call); `bf16` and `fp8` (the default) are smaller but round the hidden state. Results: [`bench/olmoe/2026-11-05.md`](../bench/olmoe/2026-11-05.md). A tiny random OLMoE (`ref_olmoe.py --tiny`, in `tests/golden/`) runs the same forward pass in `zig build test`, and so does a tiny Qwen3-MoE with YaRN (`ref_qwen3.py --tiny`).

Token generation (TASK-24): add `--generate 64` to the reference command to also record 64 tokens of transformers' greedy `generate()` after each prompt (written to `--out` with the rest), then add `--generate 64` to any `olmoe` command. After the prompt check, it generates one token at a time with a KV cache and compares every token, and the logits it was picked from, with the reference. `--temperature T`, `--top-k K`, and `--seed S` sample instead (repeatable for a given seed, so not compared). `--minutes 10` then repeats the generation for 10 minutes and fails if any round's tokens or logits change or resident memory grows by 1% or more after the first round. It names each round that differs (prompt, first differing token, largest logit difference), and each process checks at exit that its weights in memory are unchanged.

Lookahead predictor (TASK-11): add `--lookahead 2` to any `olmoe` command. After each layer's attention, rank 0 guesses the experts two layers ahead (the first layers from the token embeddings) by applying that later layer's own norm and router, keeps the top 8 + `--extra` (default 2), and sends rank 1 a prefetch hint for its share. It reports recall (the share of chosen experts that were guessed), the time spent guessing as a share of layer time, and runs every prompt and generation a second time with the predictor off to check the output is bit-identical. Rank 1 reports how long before each layer's first expert call its hint arrived. `--lookahead 1` guesses one layer ahead. `tools/ref/lookahead_study.py` measures the same recall in transformers over text files, for example WikiText-2 and HumanEval.

Expert cache (TASK-11): add `--cache N` to hold each process's experts in N slots (about 6.7 MB each for OLMoE) instead of loading them all at startup. A worker thread reads experts from the model file through the io_uring reader when a prefetch hint names them, and on demand when a call needs one that is not loaded; each expert's CRC32C is checked the first time it is read. `--pin L` keeps layers below L loaded (default 1, since layer 0 is guessed worst). Each run reports how expert calls found their expert: already loaded, loaded by a hint in time, hinted but still loading, or missed. On a GPU that reads host memory in place (GB10) the cached experts run on the GPU; on a discrete GPU they run on the CPU.

Miss handling (TASK-14): `--loaders N` sets how many threads read experts for the cache, each with its own reader (default 2). A hint whose layer has already started is dropped instead of loaded, and an expert that is not loaded when a call needs it is loaded ahead of every hint. Both ranks run the calls whose experts are ready first and the rest as their experts arrive; results are still added in a fixed order, so the output does not change. `--mispredict 0.1` replaces 10% of the guesses with wrong experts, to test all of this. With the predictor on, `--minutes M` checks every round's tokens and logits against a round run with the predictor off.

Swarm tracker (TASK-09): `tracker serve` (built by `zig build install`, Linux) keeps the live list of engine processes on TCP port 47300. Each process sends a heartbeat every 250 ms, and every change goes out to all of them at once. A process is marked unhealthy when its connection closes without a goodbye (killed) or after 1 s without a heartbeat (stopped or hung). Add `--tracker IP[:PORT]` to `olmoe expert` (or `rdma-expert`) to announce its address and its experts, and to `olmoe router` (or `rdma-router`) in place of `--host`. The router then waits for a healthy expert process, takes that process's experts as the placement, and stops with `error.PeerUnhealthy` instead of routing to it once the tracker marks it unhealthy. `--advertise IP` sets the address an expert announces; by default it is the one the tracker sees. `tracker watch IP` prints the map on every change, and `tracker bench IP --peers 16` measures how long joins and failures take to reach every peer:

```bash
zig-out/bin/tracker serve
```

```bash
zig-out/bin/olmoe rdma-expert ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf --port 7480 --tracker <tracker IPv4> --gpu --payload f32
```

```bash
zig-out/bin/olmoe rdma-router ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf --tracker <tracker IPv4> --golden ~/models/olmoe_forward.tfref --gpu --payload f32
```

Two-process dispatch and combine check (TASK-02), over shared memory, or over TCP between two machines (start the expert first):

```bash
zig build install -Doptimize=ReleaseFast && zig-out/bin/dispatch shm
```

```bash
zig-out/bin/dispatch expert --port 7470
```

```bash
zig-out/bin/dispatch router --host <expert IPv4> --port 7470
```

The PoC check: add `--moe` to any of the three commands above to run the tiny MoE split across the two processes and compare against the PyTorch goldens (prints PASS, or exits non-zero).

```bash
zig-out/bin/dispatch shm --moe
```

RDMA transport (TASK-07, Linux with an RDMA device and libibverbs): build with `-Drdma=true`, then run the same checks over RoCE v2. Plain `rdma` runs both ranks on one machine through the NIC (HCA loopback); `rdma-expert` and `rdma-router` run them separately, swapping queue-pair details over TCP on `--port` first. `--dev` picks the device (default: the first), `--gid` the GID index (default: the first RoCE v2 IPv4 GID).

```bash
zig build install -Doptimize=ReleaseFast -Drdma=true -Dcpu=native && zig-out/bin/dispatch rdma --dev roceP2p1s0f0 --moe
```

`--latency` (one-way latency, comparable to `ib_write_lat`) and `--oneway` (one-direction bandwidth, comparable to `ib_write_bw`) work on every backend in place of `--moe`. Results: [`bench/transport/2026-11-05.md`](../bench/transport/2026-11-05.md).

```bash
zig build run
```

Cross-compile for the DGX Sparks:

```bash
zig build -Dtarget=aarch64-linux-gnu -Dcpu=cortex_a78
```

`-Dcpu=cortex_a78` keeps the ARMv8 CRC32C instructions (GB10's cores support them); without it the baseline target falls back to software CRC, about 14x slower per token.

## Engine on Windows 11 (native)

`dispatch.exe` and `tensorfold.exe` also build natively for Windows 11 x86_64, enough to run the PoC check without WSL2. From PowerShell (5.1 or 7) in the repo root:

```powershell
.\build.ps1
```

It finds Zig 0.17.0 on PATH or in `~\tools\zig` (installing it there with `tools\setup\install-zig.ps1`, checksum verified, no admin, if missing), runs the unit tests, builds ReleaseFast into `zig-out\bin`, and runs the dispatch and `--moe` checks over shared memory and TCP loopback. `-SkipChecks` stops after the build; `-Package` also writes a zip and its SHA-256 to `dist\`. If PowerShell refuses to run scripts, use `powershell -ExecutionPolicy Bypass -File .\build.ps1`.

```powershell
.\zig-out\bin\dispatch.exe shm --moe
```

Windows has no `fork`, so in `shm` mode the program starts a second copy of itself for rank 1, which maps the same named shared memory. TCP mode is the same as on Linux, and Windows and Linux processes can be each other's peer. The io_uring reader, the toolchain gate, CUDA, and RDMA stay Linux only.

## Lab machine access

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

The `admin` account is shared, so all project files on a Spark live in `~/lukehinojosa/` and are activated with `source ~/lukehinojosa/env.sh`.
