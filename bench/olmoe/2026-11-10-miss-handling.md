# OLMoE-1B-7B miss handling (2026-11-10)

TASK-14: what the expert cache does when the lookahead guesses wrong or a load is late, measured with 10% of guesses forced wrong on the desktop and over RDMA on `gx10-03`, plus a throughput sweep over cache size, loader threads, and hint count.

## What changed

- **Several loaders.** `--loaders N` (default 2) starts N threads, each with its own io_uring reader on the model file. They share one job queue: experts a call is waiting for first, then hints.
- **Stale hints are dropped.** Each hint carries the position (token and layer) it is for. A loader that picks up a hint whose layer has already started drops it instead of reading it.
- **Misses go first.** A call that needs an expert that is not loaded asks for it on a queue the loaders always empty before the hint queue. If every slot is busy, the request waits in a retry slot rather than being lost. An expert loaded on demand is kept ahead of hinted ones until its call has used it.
- **Run what is ready.** Both ranks run the calls whose experts are loaded first and the rest as their experts arrive. Rank 0 writes each expert's result into its own row and adds the rows in a fixed order, so the sum is the same bits whatever order the experts finish in. Rank 1 sets a waiting call aside (up to 64 tokens' worth) and answers later calls in the meantime; rank 0 matches the answers to their calls.
- **Likely guesses first.** Hints are sent in order of how many tokens in the batch voted for that expert.
- **Forced mispredictions.** `--mispredict P` replaces each guess, with probability P, by a random expert not already guessed for that token. With the predictor on, `--minutes M` first runs a round with the predictor off and checks that every later round has the same tokens and the same logits (Wyhash of every logit), not only the same tokens.

**Not done: reading experts from a peer over RDMA READ.** The ranks own disjoint sets of experts, so no peer ever holds an expert the other is missing; there is nothing to read. It only makes sense once experts are replicated across processes (TASK-15/16).

## Checks with 10% of guesses forced wrong

| Where | Setup | Result |
|---|---|---|
| Desktop, one process | CPU experts, 256 slots, `--lookahead 2 --mispredict 0.1` | recall in generation 75.7% (83.6% without forcing), 3,650 guesses forced wrong; output bit-identical with the predictor on vs off; all 192 tokens match `generate()` |
| Desktop, two processes (shared memory) | 160 slots per rank, 2-minute soak | every round identical to the predictor-off round; memory +0.00%; rank 1 set aside 23,251 calls while it ran others |
| `gx10-03`, two processes (RDMA) | GPU experts, 256 slots, 2 loaders | bit-identical; all tokens match `generate()`; 12.5 tokens/s in generation; 588 stale hints dropped on rank 0 and 916 on rank 1; hints arrived 10.3 ms before their layer on average |

## 30-minute soak (`gx10-03`, RDMA)

`olmoe rdma MODEL --golden olmoe_generate.tfref --payload f32 --gpu --generate 64 --cache 256 --lookahead 2 --extra 0 --loaders 2 --mispredict 0.1 --minutes 30`, GPU experts, run Nov 10 to 11 with the co-tenant using about 106 of 121 GB.

- **PASS.** 24,576 tokens in 30.1 minutes (128 rounds, 13.6 tokens/s on average; the last rounds ran at 12.7 to 13.8). Every round's tokens and logits were identical to a round run with the predictor off, and the prompt and generation checks against transformers passed first.
- **Memory:** rank 0 at 2,444 MB after round 1 and at most 2,444 MB (+0.00%); peak resident set 2.40 GB. Rank 1 ended at 1,837 MB.
- **Forced mispredictions:** 380,307 guesses replaced; recall at H=2 with the top 8 guesses was 69.3%. Guessing took 0.42% of layer time.
- **Cache, rank 0** (1,954,866 calls): 71.7% already loaded, 17.1% loaded by a hint in time, 2.2% hinted but still loading, 9.0% missed. 218,514 calls were set aside while others ran, 31,025 stale hints were dropped, and 155,607 hinted experts were evicted unused. 696,060 loads (4.65 TB, 3.32 ms each). Rank 1's figures were within a point.
- **Hints:** rank 1 saw 395,820 layer starts, each after its hint, by 8.4 ms on average (0.25 ms at least).

## Throughput sweep (`gx10-03`)

Two processes over RDMA, GPU experts, F32 payload, 2-minute soaks of the three prompts' 64-token generations (6 to 10 rounds each), no forced mispredictions. The co-tenant used about 106 of 121 GB throughout. Every run was identical in every round to its reference (the predictor-off round when hints were on), and resident memory stayed flat (+0.00%). TASK-11's numbers, from before this task (one loader, no stale-hint drops, calls run strictly in order), are in the first column.

| Slots per rank | Hints | TASK-11 | 1 loader | 2 loaders | 4 loaders |
|---|---|---|---|---|---|
| 128 (25%) | off | 8.0 | 8.8 | | **10.2** |
| 128 | top 8 + 2 | 7.9 | 9.9 | 9.8 | 8.6 |
| 128 | top 8 | 9.0 | 9.5 | 9.8 | 9.2 |
| 256 (50%) | off | 12.3 | 13.2 | | 14.5 |
| 256 | top 8 + 2 | 12.6 | 14.3 | 13.9 | 13.2 |
| 256 | top 8 | 13.1 | 14.0 | **14.6** | 14.2 |

Tokens/s on rank 0. With every expert loaded the same setup runs at 20.4 tokens/s.

Rank 0's cache in some of these runs:

| Slots | Hints | Loaders | Already loaded | Hint in time | Hint still loading | Missed | Calls set aside | Load time | Stale hints dropped | Hinted, evicted unused |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | off | 1 | 43.0% | | | 56.8% | 51,797 | 2.02 ms | | |
| 128 | top 8 | 1 | 49.2% | 15.7% | 7.0% | 28.2% | 37,290 | 2.23 ms | 16,598 | 5,503 |
| 128 | top 8 | 2 | 49.6% | 22.6% | 7.3% | 20.5% | 33,650 | 3.68 ms | 7,864 | 9,493 |
| 128 | top 8 | 4 | 49.4% | 27.7% | 4.6% | 18.4% | 24,346 | 6.16 ms | 1,265 | 10,423 |
| 256 | off | 4 | 73.0% | | | 25.5% | 40,880 | 3.76 ms | | |
| 256 | top 8 | 2 | 74.0% | 15.3% | 2.2% | 8.6% | 17,929 | 3.23 ms | 2,233 | 6,840 |

What this shows:

- **Every setup got faster**: the best run of each row is 9 to 28% above TASK-11's, including the runs without hints: running ready experts while others load helps on its own.
- **The drive is still the limit.** One expert takes about 2 ms to read with one loader, 3.7 with two, and 6 to 7 with four, so each rank reads about 3.3, 3.6, and 3.9 GB/s. With both ranks on one drive, that is close to its 7 GB/s ceiling at two loaders already. More loaders turn more hints into hits (27.7% in time with four, against 15.7% with one) and drop fewer as stale, but the extra reads crowd out the misses, so throughput stops rising.
- **Hints and plain caching are now close.** At 256 slots, no hints with four loaders (14.5) matches the best hinted run (14.6). Hints still cut misses by about two thirds; with a drive that is already busy, that mostly trades miss reads for hint reads. They should pay off more once each rank has its own drive (experts on two machines).
- Run to run, the same setup varies by about one round (roughly 1 token/s here), so differences under that are noise. Two loaders is never far from the best, so it stays the default.

Rank 1 reports no time with nothing to run, because it never stops to wait for a load: it sets the call aside and keeps receiving. Its other figures were within 3 points of rank 0's.

## Acceptance (TASK-14)

| Criterion | Target | Result |
|---|---|---|
| Runs under forced 10% mispredictions | 30 minutes, tokens produced continuously | 30.1 minutes, 128 rounds, 13.6 tokens/s |
| No crash or deadlock | | exited 0, every round finished |
| No memory growth | | +0.00% after round 1 |
| Output vs a no-speculation run | bit-identical | tokens and logits identical to the predictor-off round in all 128 rounds |

## Not yet measured or done

- Reading experts from a peer over RDMA READ (see above; needs replicated experts).
- Experts on two machines, each with its own drive, where the extra loaders and hints should help most.
- Cancelling a read that has already started: a loader drops a stale hint only before it starts reading.
