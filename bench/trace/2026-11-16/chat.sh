#!/bin/bash
# TASK-33 chat replays: Qwen3-30B-A3B split across the three DGX Sparks over RDMA, then the same plus
# the desktop's CPU over TCP. Rank 0 on gx10-02 serves the chat (tools/ref/record_chat.py) and records it.
# The Sparks are cabled as a hub: gx10-02's rocep1s0f0 to gx10-03, its rocep1s0f1 to gx10-01.
# Run from Git Bash on the desktop with FWD_TO set to the WSL2 VM's address (`wsl hostname -I`):
#   chat.sh [three] [desktop]  (both by default)
out="$(cd "$(dirname "$0")" && pwd)"
wout="/mnt/$(echo "${out:1:1}" | tr A-Z a-z)${out:2}"
DESK=184.171.155.166
G2="ssh -o HostName=100.122.124.37 gx10-02"
E='source ~/lukehinojosa/env.sh; cd ~/lukehinojosa'
M=models/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf
O2=src/engine/zig-out-gen/bin/olmoe
O=bin/olmoe

# expert HOST-IP DEV RANK RANKS PORT NAME: an RDMA expert on another Spark, started from gx10-02.
expert() {
    $G2 -n "ssh -n admin@$1 'cd ~/lukehinojosa; nohup sh -c \"echo \\\$\\\$ > /tmp/lh_chat.pid; exec $O rdma-expert $M --port $5 --dev $2 --rank $3 --ranks $4 --gpu\" > /tmp/lh_chat_$6.log 2>&1 < /dev/null &'"
}

# record NAME RANKS PEERS...: serves on gx10-02 with --record, holds the chat, then stops everything.
record() {
    local name=$1 ranks=$2
    shift 2
    $G2 "$E; nohup $O2 rdma-router $M $* --gpu --serve 127.0.0.1:8092 --ctx 8192 --record /tmp/lh_chat_$name.jsonl > /tmp/lh_chat_$name.r0.log 2>&1 < /dev/null & echo \$! > /tmp/lh_chat_r0.pid"
    $G2 "for i in \$(seq 120); do curl -s -o /dev/null http://127.0.0.1:8092/ && break; sleep 1; done"
    $G2 "cd ~/lukehinojosa/src/engine && python3 tools/ref/record_chat.py --base http://127.0.0.1:8092 --thinking --max-tokens 500" | tee "$out/chat-$name.txt"
    sleep 3
    # SIGTERM: a background job of a non-interactive shell ignores SIGINT. The experts stop after rank 0 is gone.
    $G2 'p=$(cat /tmp/lh_chat_r0.pid); kill -TERM $p; while kill -0 $p 2>/dev/null; do sleep 0.5; done'
    for ip in 192.168.197.13 192.168.177.11
    do
        $G2 -n "ssh -n admin@$ip 'kill \$(cat /tmp/lh_chat.pid) 2>/dev/null; sleep 1; rm -f /tmp/lh_chat.pid'"
    done
    scp -q -o HostName=100.122.124.37 gx10-02:/tmp/lh_chat_$name.jsonl "$out/chat-$name.jsonl"
    scp -q -o HostName=100.122.124.37 gx10-02:/tmp/lh_chat_$name.r0.log "$out/chat-$name.rank0-gx10-02.log"
    $G2 "scp -q admin@192.168.197.13:/tmp/lh_chat_$name-r1.log /tmp/ 2>/dev/null; scp -q admin@192.168.177.11:/tmp/lh_chat_$name-r2.log /tmp/ 2>/dev/null; true"
    scp -q -o HostName=100.122.124.37 "gx10-02:/tmp/lh_chat_$name-r1.log" "$out/chat-$name.rank1-gx10-03.log" 2>/dev/null
    scp -q -o HostName=100.122.124.37 "gx10-02:/tmp/lh_chat_$name-r2.log" "$out/chat-$name.rank2-gx10-01.log" 2>/dev/null
    gzip -f "$out/chat-$name.jsonl"
}

# Three Sparks over RDMA.
three() {
    local P=$(( 47900 + RANDOM % 40 ))
    expert 192.168.197.13 rocep1s0f1 1 3 $P three-sparks-r1
    expert 192.168.177.11 rocep1s0f0 2 3 $P three-sparks-r2
    sleep 3
    record three-sparks 3 --peer rdma:192.168.197.13:$P@rocep1s0f0 --peer rdma:192.168.177.11:$P@rocep1s0f1
}

# The same plus the desktop's CPU over TCP: one router with RDMA and TCP links.
desktop() {
    local P=$(( 47950 + RANDOM % 40 )) D=$(( 7800 + RANDOM % 50 ))
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "cd ~ && ~/tfs-scratch/n33-cpu/bin/olmoe expert ~/$M --port $D --rank 3 --ranks 4 > '$wout/chat-sparks-desktop.rank3-desktop.log' 2>&1" &
    local ep=$!
    python "$out/../../olmoe/2026-11-11-cpu-expert/fwd.py" $((D + 50)) $D $FWD_TO > /dev/null 2>&1 &
    local fp=$!
    expert 192.168.197.13 rocep1s0f1 1 4 $P sparks-desktop-r1
    expert 192.168.177.11 rocep1s0f0 2 4 $P sparks-desktop-r2
    sleep 3
    record sparks-desktop 4 --peer rdma:192.168.197.13:$P@rocep1s0f0 --peer rdma:192.168.177.11:$P@rocep1s0f1 --peer tcp:$DESK:$((D + 50))
    wait $ep
    kill $fp 2>/dev/null
}

for s in "${@:-three desktop}"
do
    for t in $s
    do
        echo "== $t"
        $t
    done
done
$G2 'rm -f /tmp/lh_chat_*'
echo CHAT_DONE
