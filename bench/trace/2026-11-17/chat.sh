#!/bin/bash
# TASK-31 chat replay: Qwen3.8-Flash-Next on gx10-02 alone, experts streamed from NVMe through a 12,000-slot cache.
# Rank 0 serves the scripted chat (tools/ref/record_chat.py) and records it. Run on gx10-02: chat.sh OUT_DIR
out="${1:-$(cd "$(dirname "$0")" && pwd)}"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O=src/engine/zig-out-q38k/bin/olmoe
M=models/qwen38-flash-next/Qwen3.8-Flash-Next-Q8_0-00002-of-00002.gguf
$O local $M --gpu --cache 12000 --max-step 2048 --serve 127.0.0.1:8093 --ctx 8192 --record "$out/chat-qwen38-one-spark.jsonl" > "$out/chat-qwen38-one-spark.log" 2>&1 &
pid=$!
for i in $(seq 300)
do
    curl -s -o /dev/null http://127.0.0.1:8093/ && break
    sleep 1
done
python3 src/engine/tools/ref/record_chat.py --base http://127.0.0.1:8093 --thinking --max-tokens 500 | tee "$out/chat-qwen38-one-spark.txt"
sleep 3
# SIGTERM: a background job of a non-interactive shell ignores SIGINT.
kill -TERM $pid
while kill -0 $pid 2>/dev/null
do
    sleep 0.5
done
gzip -f "$out/chat-qwen38-one-spark.jsonl"
echo CHAT_DONE
