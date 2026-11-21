# A chat with an expert rank on the desktop's GPU under native Windows (2026-11-20)

Qwen3-30B-A3B (Q8_0) over four ranks, where rank 3 runs on the desktop's RTX 5070 Ti in native Windows, which TASK-40 made possible. `gx10-01` and `gx10-03` were busy with their owners' servers (both GPUs at 96%, 2 to 20 GB free), so the other three ranks ran on `gx10-02`. Each layer still waits on the desktop as it would with four machines, so this measures what the desktop and its link cost.

The chat is TASK-33's: `tools/ref/record_chat.py --thinking --max-tokens 500`, three turns at temperature 0.

## Setup

- **Ranks:**
  - **0, 1 and 2** on `gx10-02`: rank 0 serves the chat and records it, and ranks 1 and 2 are RDMA experts reached through `gx10-02`'s own ConnectX-7 ports.
  - **Rank 3** is the one run three ways: on `gx10-02` as a third RDMA expert (the baseline), on the desktop's GPU, and on the desktop's CPU (i7-12700K).
  - Experts are split evenly, 32 of 128 per rank.
- **Link to the desktop:** TCP on the campus path, rank 0 dialing the desktop's campus address directly: no WSL2, `fwd.py` or Tailscale. `ping` gave 2.0 ms round trips (1.6 to 2.6). The installer's firewall rule for `tfswarm.exe` let the connection in.
- **Build:** commit `23c64d3`, ReleaseFast.
  - **`gx10-02`:** `-Drdma=true -Dgpu=true`, kernel 7.0.0-1019-nvidia, driver 580.178.04.
  - **Desktop:** `.\build.ps1 -Gpu`, driver 617.42.
- **Activations:** FP8, the default.
- **Files:** the script is `2026-11-20-desktop-gpu/chat.sh`. That folder also has each run's recording (`chat-*.jsonl.gz`, playable on the live page), its transcript, and every rank's log.

**On the GPU, the desktop's experts went into device memory:** "weights copied to device memory (7.70 GB in 1.11 s)". The 16 GB card holds a quarter of the experts with room to spare.

## Results

| Rank 3 on | Decode | Per token (median) | Wait per layer | First tokens |
|---|---|---|---|---|
| `gx10-02` (RDMA, the baseline) | 14.4 to 14.7 tokens/s | 68.8 ms | 1.11 ms | 0.45, 1.84, 1.71 s |
| Desktop GPU (TCP) | **6.0 to 6.4 tokens/s** | 161.1 ms | 3.07 ms | 0.82, 3.19, 1.31 s |
| Desktop CPU (TCP) | 3.5 to 3.6 tokens/s | 282.4 ms | 6.31 ms | 4.0, 18.6, 21.3 s |

- **Decode** is the server's own figure for each reply.
- **Wait per layer** is rank 0's average time waiting for its slowest peer, from the recording.
- **Calls:** rank 3 took 26.4% of expert calls in each run (129,634 of 490,752 on the GPU run).

**The link costs about 2 ms per layer.** Moving rank 3 to the desktop's GPU added 1.96 ms to each layer's wait, about the ping's 2.0 ms. Across 48 layers that is 94 ms per token, and it takes the median token from 69 to 161 ms. The desktop's GPU work is a small part of it.

**The CPU adds about 3 ms more per layer.** That is about two expert calls per layer at about 1.4 ms each, as on Nov 16. The GPU made this run 1.7 times as fast as the CPU's.

**First tokens** suffer most on the CPU. Prefill sends each expert many tokens, and on the CPU the split runs them one call at a time (`../olmoe/2026-11-16-nodes.md`).

**Replies differ between runs.** FP8 activations round differently depending on which rank runs an expert, as in TASK-33's recordings. Every reply is coherent, and one per run hit the 500-token limit except in the baseline.

## Against earlier runs

- **Nov 16, four machines with the desktop's CPU in WSL2:** 2.0 tokens/s. That run used `fwd.py`, and `gx10-01`'s GPU was shared. Today's native CPU run gets 3.5 to 3.6 with no Spark contention, so the two are not directly comparable.
- **Three Sparks alone over RDMA, `gx10-01` free:** 27.5 tokens/s.
- **Why today's baseline is only 14.5 tokens/s:** all four of its ranks shared one GB10 GPU.
- **With the real four machines:** a run with the desktop's GPU would still wait about 3 ms per layer on the desktop. So it would land near today's 6 tokens/s, not near 27.5.

## Not done

- **The real four-machine run:** `gx10-01` and `gx10-03` were busy. It needs a window agreed with their owners.
- **Placement that keeps calls off the slow link:** that is TASK-56's placement by use. With an even split, every layer pays the round trip.
