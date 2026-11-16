"""Check a TASK-16 recording against the engine's own counters.

    python tools/ref/check_trace.py FILE.jsonl[.gz|.xz] ...

For each recording: the steps must be gapless with none dropped, every token
must have top_k experts in every layer, and each rank's expert calls summed from
the frames must equal the engine's counters over the same steps.
"""
import gzip
import json
import lzma
import sys


def check(path):
    opener = gzip.open if path.endswith(".gz") else lzma.open if path.endswith(".xz") else open
    hello = None
    end = None
    steps = 0
    tokens = 0
    seq_next = None
    gaps = 0
    first_engine = None
    last_engine = None
    first_drawn = None
    drawn = None
    chats = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o.get("t")
            if hello is None:
                hello = o
                ranks = len(o["ranks"])
                owner = o["owner"]
                drawn = [0] * ranks
                continue
            if t == "chat":
                chats += 1
            elif t == "end":
                end = o
            elif t == "step":
                if seq_next is not None and o["seq"] != seq_next:
                    gaps += 1
                seq_next = o["seq"] + 1
                x = o["x"]
                this = [0] * ranks
                for j in range(0, len(x), 3):
                    this[min(owner[x[j + 1]], ranks - 1)] += x[j + 2]
                for r in range(ranks):
                    drawn[r] += this[r]
                if first_engine is None:
                    first_engine = o["engine"]
                    first_drawn = this
                last_engine = o["engine"]
                steps += 1
                tokens += o["n"]
    engine = []
    for r in range(len(drawn)):
        if first_engine[r] is None:
            engine.append(None)
        else:
            engine.append(last_engine[r] - (first_engine[r] - first_drawn[r]))
    shape_ok = sum(drawn) == tokens * hello["layers"] * hello["top_k"]
    counts_ok = all(e is None or e == d for e, d in zip(engine, drawn))
    dropped = end["dropped"] if end else None
    ok = gaps == 0 and shape_ok and counts_ok and dropped in (0, None)
    print(f"{path}: {hello['model']} on {hello['ranks']} ({hello['transport']}), {steps} steps, {tokens} tokens, "
          f"{chats} requests; calls by rank {drawn}, engine {engine}; gaps {gaps}, "
          f"end {'yes' if end else 'no'}, dropped {dropped}: {'ok' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    results = [check(p) for p in sys.argv[1:]]
    sys.exit(0 if results and all(results) else 1)
