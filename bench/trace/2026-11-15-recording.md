# Telemetry recording and replay (2026-11-15)

TASK-16. `olmoe <mode> MODEL --record FILE` writes every engine step's expert choices to FILE, in any mode: the server, the long-context run, and the golden checks with their soak. The built-in page (`web/live.html`) replays a recording with its chat alongside.

## Recording

A recording is the live feed's frames (TASK-16b), one JSON object per line:

- It starts with the `hello` line, which also gives the time it was recorded.
- One `step` line follows per engine step. A prefill chunk is summarized as counts per layer and expert; each decode step is one token.
- Under `--serve`, notes from the server sit among the steps:
  - the request (`chat`);
  - each piece of reply text (`text`), after the step that produced its token;
  - how the reply finished (`done`).
- An `end` line closes a run that finished on its own, with its step and drop counts. A server stopped with a signal has no `end` line; its last step is its end.

The format is in `docs/telemetry-schema.md`.

How it stays off the token path:

- `olmoe.step` now calls an optional begin and end hook on `Moe`. The telemetry tap sets these hooks, so every path sees each step without changes at the call sites.
- The tap counts the router's own choices on the engine thread and copies them into the feed's ring, which the engine never waits on.
- The broadcaster thread that serves viewers also writes the file. A recording counts as a viewer, so the tap counts every step while it records.
- Notes from the engine thread carry the number of steps published when they were made, so the writer puts each one after exactly those steps.
- A full ring drops steps instead of blocking. The `end` line counts them, and gaps in `seq` show them.

Checked with `tools/ref/check_record.py`:

- It runs `--context 512 --generate 256` once with `--save` as the reference.
- Then it alternates three runs without `--record` and three with it, each with `--compare` against the reference, so every run must give the same tokens and logits bit for bit.
- Each recording must hold every step, with none dropped.

Build: commit `3a9f245` plus this change, ReleaseFast, `-Drdma=true -Dgpu=true`.

| Setup | Decode without recording | With recording | Tokens and logits | Recording |
|---|---|---|---|---|
| Qwen3-30B-A3B, `local` on `gx10-02`, `--gpu` | 36.58 tokens/s | 36.65 (+0.2%) | identical to the reference in all 6 runs | 256 steps, 0.92 MB, none dropped |
| OLMoE-1B-7B, `rdma-router` on `gx10-02` + `rdma-expert` on `gx10-03`, `--gpu --payload f32 --chunk 64` | 95.95 tokens/s | 96.10 (+0.2%) | identical to the reference in all 6 runs | 263 steps, 0.34 MB, none dropped |

The soak-style run used the same split and the TASK-15 run's shape (golden prompts, `--generate 64 --minutes 3`):

- **With `--record`:** 93.3 tokens/s, 88 rounds.
- **Without:** 92.0 tokens/s, 87 rounds.
- **Both:** every round was identical to the first, and resident memory stayed flat.
- **The recording:** 17,091 steps, none dropped (21 MB, 3.7 MB gzipped).

## Recordings checked against the engine

For each recording, the expert calls summed from its frames, by rank, equal the engine's own counters over the same steps (`SplitMoe`'s local and remote calls for a split, `LocalExperts.calls` for one process). The steps were gapless in every recording.

| Recording | Steps | Tokens | Requests | Expert calls, rank 0 / rank 1 (frames = engine) |
|---|---|---|---|---|
| Chat, OLMoE split over RDMA (`chat-olmoe-two-sparks-rdma.jsonl.gz`) | 342 | 429 | 3 | 28,353 / 26,559 |
| Chat, Qwen3-30B-A3B on `gx10-02` with reasoning (`chat-qwen3-one-spark.jsonl.gz`) | 1,129 | 1,329 | 3 | 510,336 |
| Soak, 3 minutes, OLMoE split over RDMA | 17,091 | 20,691 | 0 | 1,351,583 / 1,296,865 |

The two chat recordings are in `bench/trace/2026-11-15/`. They came from `tools/ref/record_chat.py`, a scripted three-turn conversation at temperature 0, with up to 160 tokens per reply for OLMoE and 500 for Qwen3 with thinking on. The soak recording is not kept. TASK-15's one-hour soak will be recorded and archived with that run.

## Replay

The page replays a recording from "Replay a recording" (or a file dropped on it), or from `?trace=URL`, the way TASK-18's static site will load one. Gzipped files are read directly.

- **Playback:** play, pause, a speed from 0.25x to 10x, and a scrubber.
- **Timing:** recorded time, with idle stretches between requests cut to one second.
- **Chat:** messages and replies appear as their steps play, with Qwen3's reasoning folded apart.
- **Stats:** the same as live, including each rank's expert calls checked against the engine's counters carried in the frames.

In the browser pane on the desktop (Chromium, 144 Hz display):

- **Frame rate:** the 3-minute soak (17,091 steps) loaded in 178 ms and replayed at 144.8 frames per second, 6.9 ms per frame at p50 and 7.1 at p99. Seeking from 60 s to 170 s took 56 ms.
- **Counts:** at every position checked, in all three recordings, the page's per-rank counts matched the engine's.
- **Malformed files:** an empty file, a file that is not JSON, a hello whose owner list does not match its experts, and a file with no steps each showed a message saying why instead of a broken page. A file with three corrupt lines and a cut-off last line replayed with the bad lines counted as rejected and the cut-off noted.
- **Live mode:** it still works after the change: a chat on the desktop streamed with the diagram updating and its counts matching.

Notes:

- Logits were compared bit for bit through `--compare`, which covers what TASK-16b could not check through the server.
- Packet timings are the per-step round trip already in the live feed: `rtt_us` over `rtt_layers`, timed around each layer with remote calls. Individual packets are not recorded.
- The second rank shows as the address dialed (`192.168.197.13`) unless the run uses the tracker.
