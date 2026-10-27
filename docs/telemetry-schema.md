# Telemetry Trace Schema v0

Status: **v0, PoC** (TASK-04). The WebGL inspector plays these traces; the engine starts emitting them in TASK-16. Until then, traces come from the synthetic generator in [`web/src/trace.js`](../web/src/trace.js), whose timing uses the measured hardware numbers in `bench/`.

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
| `links` | Node pairs (indices into `nodes`) and the transport between them: `"rdma"`, `"tcp"`, or `"shm"`. |
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
