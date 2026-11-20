# Telemetry Trace Schema v0

Status: **v0, PoC** (TASK-04). The WebGL inspector plays these traces, which come from the synthetic generator in [`web/src/trace.js`](../web/src/trace.js), whose timing uses the measured hardware numbers in `bench/`.

The engine does not write this format. What it records (TASK-16) and streams live (TASK-16b) is the per-step feed described in [Live feed and recordings](#live-feed-and-recordings-tensorfoldlive-v0) below, which the 2D view in `web/live.html` plays. Per-packet events like `"D"` and `"C"` would be one event per token, layer and expert: about 786,000 for a single 2,048-token Qwen3 prefill chunk.

A trace is one JSON document. Events are compact positional arrays, because a single forward pass of a real model produces tens of thousands of them.

## Document

```json
{
  "schema": "tensorfold.trace",
  "version": 0,
  "source": "synthetic",
  "time_unit": "ns",
  "model": { "name": "OLMoE-1B-7B", "layers": 16, "experts": 64, "top_k": 8, "d_model": 2048, "d_ff": 1024 },
  "nodes": [ { "id": "gx10-03", "label": "gx10-03", "kind": "gb10" } ],
  "links": [ { "a": 0, "b": 1, "transport": "rdma" } ],
  "processes": [ { "id": 0, "node": 0, "rank": 0 } ],
  "expert_owner": [ 0, 1, 2, 0, 1, 2 ],
  "batch_tokens": 8,
  "generator": { "preset": "sparks", "seed": 42 },
  "events": [ [0, "L", 0, 0], [0, "D", 3260, 0, 0, 0, 17, 0, 2, 7552] ]
}
```

| Field | Meaning |
|---|---|
| `schema`, `version` | Always `"tensorfold.trace"` and `0` for this spec. Readers reject anything else. |
| `source` | `"synthetic"` (generated) or `"engine"` (recorded). The inspector labels the view accordingly. |
| `time_unit` | Always `"ns"`. Event times are nanoseconds since trace start. |
| `model` | Shape of the model being run. `expert_owner` and expert indices refer to it. |
| `nodes` | Machines. `kind` is `"gb10"` or `"desktop"`. Referenced by index. |
| `links` | Node pairs (indices into `nodes`) and the transport between them: `"rdma"`, `"tcp"` (over a relay), `"tcp-direct"`, `"tcp-tailscale"`, or `"shm"`. |
| `processes` | Engine processes (ranks) and the node each runs on. Events name processes, not nodes. |
| `expert_owner` | For each expert index, the process that holds it. v0 assumes the same placement in every layer. |
| `batch_tokens` | Tokens per batch. |
| `generator` | Optional, free-form: how a synthetic trace was made. |
| `events` | Sorted by time (ascending). |

## Events

Every event starts with `[t, kind, ...]`, where `t` is the start time in ns. Fields are in this exact order:

| Kind | Name | Fields after `t, kind` |
|---|---|---|
| `"L"` | Layer begin | `layer, batch` |
| `"D"` | Dispatch packet (token to expert) | `dur, batch, token, layer, expert, src_proc, dst_proc, bytes` |
| `"X"` | Expert compute | `dur, batch, layer, expert, proc, status` |
| `"C"` | Combine packet (expert result back) | `dur, batch, token, layer, expert, src_proc, dst_proc, bytes` |
| `"F"` | Expert fetch from NVMe | `dur, layer, expert, proc, bytes, reason` |

- `dur` is in ns. A dispatch or combine with `src_proc == dst_proc` is local and has `dur` 0.
- `token` is the index within the batch; `batch` counts up from 0.
- `bytes` for packets is the wire packet size (7,552 in [wire format v0](wire-format.md)).
- `status` on `"X"`: `0` the expert was already resident, `1` prefetched by the lookahead router, `2` a miss (the compute waited on an NVMe fetch).
- `reason` on `"F"`: `0` lookahead prefetch, `1` demand fetch after a miss.
- One `"X"` covers all of a batch's tokens routed to that expert in that layer (experts run as grouped GEMMs).

Readers must ignore event kinds they do not know, so later versions can add kinds without breaking v0 players. Changing the fields of an existing kind requires a new `version`.

## Mapping to the inspector

| Event | Rendering |
|---|---|
| `"L"` | Layer counter in the stats panel |
| `"D"`, `"C"` | A packet flying between the two processes' nodes, colored by link transport; local ones are counted, not drawn |
| `"X"` | The expert cube flashes: blue resident, green prefetched, red miss |
| `"F"` | Counted (NVMe traffic) |

Playback maps trace time to screen time so that the median layer takes about 1.5 seconds at 1x speed, since real layers last microseconds. Packet flights are drawn for at least 250 ms so they stay visible.

## Live feed and recordings (`tensorfold.live` v0)

Written by `src/telemetry.zig`. The live feed at `/telemetry` (TASK-16b) sends each object as a WebSocket text frame. A recording (`tfswarm ... --record FILE`, TASK-16) holds the same objects, one per line, and may be gzipped. Times (`ms`) are milliseconds on the engine machine's monotonic clock, so only differences between them mean anything.

| Line | Fields |
|---|---|
| `hello` (first) | `schema` (`"tensorfold.live"`), `version` (`0`), `model`, `layers`, `experts`, `top_k`, `transport`, `ranks` (one name per engine process), `owner` (for each expert, the rank holding it in every layer), `chunks` (packets per remote expert call each way, 0 if nothing is remote), `cache_ranks` (the ranks whose experts come through an expert cache, so the page draws a drive there; added for TASK-41, absent in older recordings), `ms`; recordings add `recorded` (Unix seconds) |
| `step` | `seq` (counts up from 0; a gap means steps were dropped), `ms` (when the step ended), `n` (tokens: a prefill chunk, or 1 when decoding), `step_us`, `layer_us` (one per layer), `x` (flat triples `layer, expert, count` for every expert that ran), `engine` (expert calls the engine itself has counted so far, by rank, or `null` where it keeps no count), `rtt_us` and `rtt_layers` (time from first dispatch to last combine, summed over the layers with remote calls so far), `cache` (`{resident, hinted, late, missed, loads}` so far, or `null` without `--cache`), `c` and `h` (below; added for TASK-41, absent in older recordings) |
| `chat` (recordings from `--serve`) | `after`, `id`, `messages` (`{role, content}`, the whole conversation the client sent), `prompt_tokens`, `max_tokens`, `temperature` |
| `text` | `after`, `channel` (`"content"` or `"reasoning"`), `s` (reply text that became visible after step `after`) |
| `done` | `after`, `id`, `finish` (`"stop"`, `"length"`, or `"client_gone"`), and the reply's `prompt_tokens`, `cached_tokens`, `completion_tokens`, `ttft_ms`, `decode_tokens_per_s` |
| `end` (last, when the run finished on its own) | `after`, `steps` (lines written), `dropped` (steps lost to a full ring), `ms` |

**Cache and prefetch fields (TASK-41).** `c` holds flat triples `layer, expert, status` for each of rank 0's expert-cache calls in the step: `1` resident (already in a slot), `2` hinted (loaded by a prefetch hint and ready before it was needed), `3` late (a hint's load still running when needed), `4` missed (loaded on demand; the token waited on the drive). It is empty without `--cache`. When one step calls the same expert of a layer more than once (prefill on the CPU), only the first call's status is kept; with `--gpu`, or when decoding, each expert of a layer is called once per step, so the statuses add up to the step's change in `cache`. `h` holds flat triples `for layer, expert, sent during layer` for each prefetch hint the lookahead router sent in the step (empty without `--lookahead`): the guesses from the token embeddings are sent during layer 0, and a guess made inside layer `L` (before its experts run) is sent during `L`. Hints name every guessed expert, including ones already in a slot or held by another rank. `h_dropped` appears when a step had more hints than fit (at most one per layer and expert). Both are recorded only while a viewer or a recording is attached.

`after` is the `seq` of the step written just before the line, or `null` before the first. Readers must ignore unknown line types and fields; changing an existing field's meaning requires a new `version`. The page checks a `step`'s experts against the hello's shape and rejects lines that do not fit, counting them instead of drawing them.
