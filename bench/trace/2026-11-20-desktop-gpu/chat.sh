#!/bin/bash
# Qwen3-30B-A3B chat over four ranks: three on gx10-02 (rank 0 and two RDMA experts through its own NIC),
# and rank 3 either on gx10-02 too, or on the desktop's RTX 5070 Ti or CPU under native Windows over TCP.
# Run from Git Bash on the desktop, in the repo, after `.\build.ps1 -Gpu`:
#   chat.sh [spark] [desktop-gpu] [desktop-cpu]  (all by default)
out="$(cd "$(dirname "$0")" && pwd)"
DESK=184.171.155.166
G2="ssh -o HostName=100.122.124.37 gx10-02"
E='source ~/lukehinojosa/env.sh; cd ~/lukehinojosa'
M=models/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf
O=zig-out-chat/bin/tfswarm
WM='D:\models\qwen3-30b-a3b\Qwen3-30B-A3B-Q8_0.gguf'
WO=zig-out/bin/tfswarm.exe
A=192.168.197.12
B=192.168.177.12

# expert NAME RANK DEV PORT: an RDMA expert on gx10-02, reached through its own NIC.
expert()
{
    $G2 "$E; nohup $O rdma-expert $M --port $4 --dev $3 --rank $2 --ranks 4 --gpu > /tmp/lh_c20_$1-r$2.log 2>&1 < /dev/null & echo \$! >> /tmp/lh_c20.pids"
}

# ready NAME N: waits until gx10-02's experts 1 to N are listening (each takes its own handshake port).
ready()
{
    $G2 "for i in \$(seq 60); do n=\$(cat /tmp/lh_c20_$1-r*.log 2>/dev/null | grep -c listening); [ \$n -ge $2 ] && break; sleep 1; done; grep -h error /tmp/lh_c20_$1-r*.log"
}

# record NAME PEERS...: serves on gx10-02 with --record, holds the chat, then stops everything there.
record()
{
    local name=$1
    shift
    $G2 "$E; nohup $O rdma-router $M $* --gpu --serve 127.0.0.1:8092 --ctx 8192 --record /tmp/lh_c20_$name.jsonl > /tmp/lh_c20_$name-r0.log 2>&1 < /dev/null & echo \$! > /tmp/lh_c20_r0.pid"
    $G2 "for i in \$(seq 300); do curl -s -o /dev/null http://127.0.0.1:8092/ && break; sleep 1; done"
    $G2 "cd ~/lukehinojosa/src-chat && python3 tools/ref/record_chat.py --base http://127.0.0.1:8092 --thinking --max-tokens 500" | tee "$out/chat-$name.txt"
    sleep 3
    # SIGTERM: a background job of a non-interactive shell ignores SIGINT. The experts stop after rank 0 is gone.
    $G2 'p=$(cat /tmp/lh_c20_r0.pid); kill -TERM $p; while kill -0 $p 2>/dev/null; do sleep 0.5; done; sleep 2; for q in $(cat /tmp/lh_c20.pids 2>/dev/null); do kill $q 2>/dev/null; done; rm -f /tmp/lh_c20.pids /tmp/lh_c20_r0.pid'
    for f in $name.jsonl $name-r0.log $name-r1.log $name-r2.log $name-r3.log
    do
        scp -q -o HostName=100.122.124.37 "gx10-02:/tmp/lh_c20_$f" "$out/chat-$f" 2>/dev/null
    done
    gzip -f "$out/chat-$name.jsonl"
}

# Every rank on gx10-02, for comparison.
spark()
{
    local P=$(( 47900 + RANDOM % 40 ))
    expert spark 1 rocep1s0f0 $P
    expert spark 2 rocep1s0f1 $((P + 1))
    expert spark 3 rocep1s0f0 $((P + 2))
    ready spark 3
    record spark --peer rdma:$A:$P@rocep1s0f0 --peer rdma:$B:$((P + 1))@rocep1s0f1 --peer rdma:$A:$((P + 2))@rocep1s0f0
}

# desk NAME [--gpu]: rank 3 on the desktop under native Windows, over the campus path.
desk()
{
    local name=$1
    shift
    local P=$(( 47950 + RANDOM % 40 )) D=$(( 7800 + RANDOM % 50 ))
    "$WO" expert "$WM" --port $D --rank 3 --ranks 4 "$@" > "$out/chat-$name-r3.log" 2>&1 &
    local ep=$!
    expert $name 1 rocep1s0f0 $P
    expert $name 2 rocep1s0f1 $((P + 1))
    ready $name 2
    for i in $(seq 300)
    do
        grep -q -i "listening" "$out/chat-$name-r3.log" && break
        sleep 1
    done
    record $name --peer rdma:$A:$P@rocep1s0f0 --peer rdma:$B:$((P + 1))@rocep1s0f1 --peer tcp:$DESK:$D
    for i in $(seq 20)
    do
        kill -0 $ep 2>/dev/null || break
        sleep 1
    done
    kill $ep 2>/dev/null
}

desktop-gpu()
{
    desk desktop-gpu --gpu
}

desktop-cpu()
{
    desk desktop-cpu
}

for s in "${@:-spark desktop-gpu desktop-cpu}"
do
    for t in $s
    do
        echo "== $t"
        $t
    done
done
$G2 'rm -f /tmp/lh_c20_*'
echo CHAT_DONE
