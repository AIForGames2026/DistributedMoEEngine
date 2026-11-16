# Server latency against olmoe directly (2026-11-15)

TASK-30 criterion (4): time to first token and decode speed through `olmoe <mode> MODEL --serve`, against `olmoe` running the same prompt tokens directly, with the live diagram's feed connected or not.

Measured with `tools/ref/bench_serve.py` on the server's machine:

- `server`: three streamed greedy requests per prompt with no viewer, then three with a viewer reading `/telemetry` (the feed the built-in page draws from), on a server started with `--no-reuse` so every request prefills its whole prompt.
- `direct`: `olmoe <mode> MODEL --context N --tokens F --generate M` three times per prompt, on the prompt token ids the server used and M = the number of tokens the server generated.

Each cell is the median of three runs. Build: commit `6d7ac83` plus the change that adds this script, ReleaseFast, `-Drdma=true -Dgpu=true`. The prompts are a short question, a medium system and user message, and a long list of station reports to summarize. Every reply was cut at 128 tokens.

Three ways to time a request:

- **olmoe directly**: from the start of prefill to the first token's logits, then the decode steps (olmoe's own report).
- **Server, engine**: the same two spans, timed by the server inside the request (`tensorfold.ttft_ms` and `decode_tokens_per_s`).
- **Server, client**: from sending the HTTP request to the first piece of reply text arriving, and the decode rate between the first and last pieces of text. This includes HTTP, the chat template, tokenizing, and streaming.

## OLMoE-1B-7B split across two Sparks over RDMA

Setup: `rdma-router` on `gx10-02` with `rdma-expert` on `gx10-03`, `--gpu --payload f32 --chunk 64`.

| Prompt | Prompt tokens | olmoe directly | Server, engine | Server, client | Server + viewer, engine | Server + viewer, client |
|---|---|---|---|---|---|---|
| short | 19 | 109 ms, 104.04 tokens/s | 107 ms, 105.61 tokens/s | 107 ms, 105.30 tokens/s | 107 ms, 105.36 tokens/s | 107 ms, 105.06 tokens/s |
| medium | 204 | 1632 ms, 99.90 tokens/s | 1580 ms, 99.48 tokens/s | 1581 ms, 99.17 tokens/s | 1628 ms, 99.60 tokens/s | 1629 ms, 99.28 tokens/s |
| long | 1346 | 10885 ms, 84.92 tokens/s | 10770 ms, 85.83 tokens/s | 10771 ms, 85.58 tokens/s | 10756 ms, 85.80 tokens/s | 10757 ms, 85.54 tokens/s |

## Qwen3-30B-A3B on one Spark

Setup: `local` on `gx10-02`, `--gpu`.

| Prompt | Prompt tokens | olmoe directly | Server, engine | Server, client | Server + viewer, engine | Server + viewer, client |
|---|---|---|---|---|---|---|
| short | 14 | 128 ms, 38.33 tokens/s | 125 ms, 38.74 tokens/s | 179 ms, 38.66 tokens/s | 125 ms, 38.40 tokens/s | 180 ms, 38.32 tokens/s |
| medium | 190 | 455 ms, 37.02 tokens/s | 410 ms, 37.16 tokens/s | 467 ms, 37.08 tokens/s | 418 ms, 37.17 tokens/s | 476 ms, 37.08 tokens/s |
| long | 1561 | 1346 ms, 35.38 tokens/s | 1349 ms, 35.67 tokens/s | 1409 ms, 35.60 tokens/s | 1335 ms, 35.67 tokens/s | 1394 ms, 35.60 tokens/s |

## What it shows

- **The server costs nothing measurable.** Decode speed inside the server is within 1.5% of olmoe's, and time to first token within 10%, in both setups. Where they differ, the server is the faster one; the widest gap is Qwen3's medium prompt (410 against 455 ms), where the server's own runs ranged from 407 to 441 ms. A fresh olmoe process likely pays some first-use cost that a warm server has already paid, but this was not isolated.
- **The HTTP and chat layers add about 1 ms.** For OLMoE, the client gets the first piece of text 1 ms after the engine produces it, and decodes at 0.3% below the engine's rate.
- **Qwen3's first visible text comes two tokens late.** Qwen3 opens every reply with `<think>` and a newline, which show no text, so its first piece of text reaches the client two decode steps (about 52 ms) after the engine's first token. Beyond those two steps, the difference is the same few milliseconds. The script's `--lead 2` keeps those two tokens out of the client's decode rate.
- **A connected diagram changes nothing.** With a viewer reading the feed, time to first token and decode speed match the runs without one (within 0.3% for decode). The tokens were identical across every repeat and both viewer settings.
- **Split prefill is the slowest part.** The split's prefill is slow: about 125 tokens per second on the long prompt, against about 1,160 for Qwen3 on one Spark. That is the engine's chunked prefill over RDMA (`--chunk 64`), the same through the server or not, and it is not this task's subject.

The desktop (WSL2, CPU, OLMoE `local`) gave the same picture in a one-run dry run: 1,035 ms to the first token at the client against 1,034 in the engine and 1,147 directly. It is not tabled, since its long prompt takes over a minute to prefill.
