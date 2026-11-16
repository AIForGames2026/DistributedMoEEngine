# Live telemetry and the built-in chat page (2026-11-15)

TASK-16b and TASK-30 criterion (3). `olmoe <mode> MODEL --serve [IP:]PORT` now also serves the chat page (`web/live.html`, embedded in the binary) at `/` and its live feed as a WebSocket at `/telemetry` (`src/telemetry.zig`). The page streams chat replies beside a 2D diagram: one grid of layers by experts per rank, each step's experts lit on the rank that ran them, the link between ranks with packets per second and the round trip per layer, and the expert calls counted by the page against the engine's own counters.

How it stays off the token path: a wrapper around the engine's `Moe` counts each layer's expert choices only while a viewer is connected, and at the end of each step copies the counts into a ring the engine thread never waits on. A separate thread turns slots into JSON frames. A viewer more than 256 KB behind (with a 128 KB kernel send buffer) has frames dropped for it, which it sees as gaps in `seq`. Feed frames are checked by the page; malformed ones are counted and ignored.

Checked with `tools/ref/check_live.py` on the server's machine: three rounds of the same greedy request (64 tokens) with no viewer, a reading viewer, and a stalled viewer (connected, never reads), then a long request (`--stall-tokens 1500`) alone and with a stalled viewer beside a reading one. Build: commit `c001e3d` plus this change, ReleaseFast, `-Drdma=true -Dgpu=true` on the Sparks.

| Setup | Decode, no viewer | Reading viewer | Stalled viewer | Tokens identical | Frames vs engine counters | Step start to frame arrival (p50 / max) | Long run with stalled viewer |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B, `rdma-router` on `gx10-02` + `rdma-expert` on `gx10-03`, `--gpu --payload f32 --chunk 64` | 105.58 tokens/s | 105.35 (-0.2%) | 105.38 (-0.2%) | yes, all 9 runs | every frame, both ranks, gapless | 15.9 / 48.9 ms | 93.94 vs 93.95 tokens/s alone (-0.0%); reader got all 1,501 frames; 854 dropped for the stalled viewer |
| Qwen3-30B-A3B, `local` on `gx10-02`, `--gpu` | 38.79 tokens/s | 38.79 (+0.0%) | 38.74 (-0.1%) | yes, all 9 runs | every frame, gapless | 31.6 / 42.4 ms | 37.26 vs 37.29 tokens/s alone (-0.1%); reader got all 526 frames; 301 dropped for the stalled viewer |
| OLMoE-1B-7B, `shm` on the desktop (WSL2, CPU), `--payload f32 --chunk 64` | 7.76 tokens/s | 7.78 (+0.3%) | 7.85 (+1.2%) | yes | every frame, both ranks | | 6.03 vs 6.02 tokens/s alone; 63 dropped for the stalled viewer |

`gx10-03` had the co-tenant's vLLM job running (about 96 GB of GPU memory, 15 GB free) during the two-Spark run; `gx10-02` was otherwise idle.

Notes:

- "Frames vs engine counters": the page's own sum of expert calls per rank, from the frames alone, equals what the engine counted: `SplitMoe`'s local and remote calls for a split, the new `LocalExperts.calls` for one process (or the expert cache's calls when `--cache` is on).
- Step start to frame arrival is measured on the server's machine, since the engine stamps frames with `CLOCK_MONOTONIC`, the clock behind Python's `time.monotonic()`. It covers the whole step plus delivery; delivery alone stayed under 40 ms.
- Logits were not compared: the server does not return them. The wrapper only reads the routing choices the engine already made, and the tokens were identical in every run.
- The stalled-viewer test must run on the server's machine. From Windows, WSL's localhost relay reads everything off the server's socket, so the server never sees the stall.
- In the browser (desktop Chrome-based pane, through an SSH tunnel to `gx10-02`) the page drew at 144 frames per second (the display's rate, 7.1 ms per frame at p99) while following the two-Spark run at 103 tokens/s. Seven hand-made malformed frames (missing fields, expert out of range, a hello whose owner list does not match) were all rejected without changing the diagram's state.
- The second rank shows as the address dialed (`192.168.197.13`) unless the run uses the tracker, which knows the node's name.
