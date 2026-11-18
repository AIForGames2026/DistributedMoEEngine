#!/bin/bash
# TASK-41 chat replays: Qwen3-30B-A3B on gx10-02 alone, experts streamed from NVMe through a 3,000-slot cache,
# once without lookahead and once with --lookahead 1. Run on gx10-02: chat.sh OUT_DIR
out="${1:-$(cd "$(dirname "$0")" && pwd)}"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O="${O:-src/engine/zig-out-t41/bin/olmoe}"
M=models/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf
for run in plain lookahead
do
    extra=""
    if [ "$run" = lookahead ]
    then
        extra="--lookahead 1"
    fi
    name="chat-qwen3-cache-$run"
    $O local $M --gpu --cache 3000 $extra --max-step 2048 --serve 127.0.0.1:8094 --ctx 8192 --record "$out/$name.jsonl" > "$out/$name.log" 2>&1 &
    pid=$!
    for i in $(seq 300)
    do
        curl -s -o /dev/null http://127.0.0.1:8094/ && break
        sleep 1
    done
    python3 src/engine/tools/ref/record_chat.py --base http://127.0.0.1:8094 --thinking --max-tokens 500 | tee "$out/$name.txt"
    sleep 3
    # SIGTERM: a background job of a non-interactive shell ignores SIGINT.
    kill -TERM $pid
    while kill -0 $pid 2>/dev/null
    do
        sleep 0.5
    done
    gzip -f "$out/$name.jsonl"
done
echo CHAT_DONE
