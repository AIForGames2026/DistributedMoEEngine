#!/bin/bash
# TASK-32 on gx10-02: GLM-5.3-Flash split across the three Sparks over RDMA, every expert in memory.
# window.sh runs it once the lab servers are stopped: run.sh SECTION (check golden chat long32 long128).
# Each section stops by STOP_AT (local HH:MM) so the window can restore the servers in time.
# RANKS=2 leaves gx10-03 out: gx10-01 holds the other half of the experts, logs go to two/.
out="$(cd "$(dirname "$0")" && pwd)"
RANKS="${RANKS:-3}"
if [ "$RANKS" = 2 ]
then
    out="$out/two"
    mkdir -p "$out"
fi
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O="${O:-src/engine/zig-out-g32g/bin/olmoe}" # O and EB override the builds on rank 0 and the expert ranks
EB="${EB:-bin/olmoe-g32}"
D=models/glm53-flash
M="${M:-$D/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf}" # M and G override for a rehearsal with the tiny model
G="${G:-$D/glm53_golden.tfref}"
STOP_AT="${STOP_AT:-05:40}"
# gx10-03 is rank 1 and gx10-01 rank 2; gx10-02's rocep1s0f0 reaches gx10-03 and rocep1s0f1 gx10-01.
R1=192.168.197.13
R2=192.168.177.11
EX='cd ~/lukehinojosa && export LD_LIBRARY_PATH=$HOME/lukehinojosa/tools/lib'
PEERS=""

# Seconds left before STOP_AT, capped at $1.
budget()
{
    local left=$(( $(date -d "$STOP_AT" +%s) - $(date +%s) ))
    if [ $left -gt $1 ]
    then
        left=$1
    fi
    echo $left
}

# experts NAME: starts the expert ranks on a fresh port and waits until each listens.
experts()
{
    local name=$1 p=$(( 48100 + RANDOM % 200 ))
    if [ "$RANKS" = 2 ]
    then
        ssh -n admin@$R2 "$EX && (nohup sh -c 'echo \$\$ > /tmp/lh_g32.pid; exec $EB rdma-expert $M --port $p --dev rocep1s0f0 --rank 1 --ranks 2 --gpu --max-step 2048' > /tmp/lh_g32_$name.log 2>&1 < /dev/null &)"
        HOSTS="$R2"
        PEERS="--peer rdma:$R2:$p@rocep1s0f1"
    else
        ssh -n admin@$R1 "$EX && (nohup sh -c 'echo \$\$ > /tmp/lh_g32.pid; exec $EB rdma-expert $M --port $p --dev rocep1s0f1 --rank 1 --ranks 3 --gpu --max-step 2048' > /tmp/lh_g32_$name.log 2>&1 < /dev/null &)"
        ssh -n admin@$R2 "$EX && (nohup sh -c 'echo \$\$ > /tmp/lh_g32.pid; exec $EB rdma-expert $M --port $p --dev rocep1s0f0 --rank 2 --ranks 3 --gpu --max-step 2048' > /tmp/lh_g32_$name.log 2>&1 < /dev/null &)"
        HOSTS="$R1 $R2"
        PEERS="--peer rdma:$R1:$p@rocep1s0f0 --peer rdma:$R2:$p@rocep1s0f1"
    fi
    for ip in $HOSTS
    do
        for _ in $(seq 30)
        do
            if ssh -n admin@$ip "grep -q listening /tmp/lh_g32_$name.log"
            then
                break
            fi
            sleep 1
        done
    done
}

# finish NAME: stops the expert ranks if still up and copies their logs next to this script.
finish()
{
    local name=$1 pairs=("$R1 rank1-gx10-03" "$R2 rank2-gx10-01")
    if [ "$RANKS" = 2 ]
    then
        pairs=("$R2 rank1-gx10-01")
    fi
    for pair in "${pairs[@]}"
    do
        set -- $pair
        ssh -n admin@$1 "kill \$(cat /tmp/lh_g32.pid) 2>/dev/null; sleep 2; kill -9 \$(cat /tmp/lh_g32.pid) 2>/dev/null; rm -f /tmp/lh_g32.pid"
        scp -q admin@$1:/tmp/lh_g32_$name.log "$out/$name.$2.log" && ssh -n admin@$1 "rm -f /tmp/lh_g32_$name.log"
    done
}

# router NAME CAP ARGS...: rank 0 on gx10-02 with its own experts in memory, stopped at CAP seconds or STOP_AT.
router()
{
    local name=$1 secs
    secs=$(budget $2)
    shift 2
    if [ "$secs" -lt 120 ]
    then
        echo "$name: skipped, $secs s left before $STOP_AT"
        return
    fi
    experts $name
    timeout --signal=TERM --kill-after=30 $secs /usr/bin/time -v $O rdma-router $M $PEERS --gpu "$@" > "$out/$name.rank0-gx10-02.log" 2>&1
    echo "$name: rank 0 exit $?"
    finish $name
    grep -E "match|calls|first token|tokens/s|prefill|decode|Elapsed|Maximum resident|error|panic" "$out/$name.rank0-gx10-02.log" | tail -25
}

# Light checks before anything loads: the links, free memory, and the staged files on each rank.
check()
{
    {
        echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
        echo "engine: $(strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ "]*g32')"
        sha256sum $D/GLM-5.3-Flash-Q4_K-0000*-of-00002.gguf
        ls -l $G $D/repo.tokens
        free -g | head -2
        for ip in $( [ "$RANKS" = 2 ] && echo $R2 || echo $R1 $R2 )
        do
            echo "== $ip"
            ssh -n admin@$ip "hostname; free -g | head -2; nvidia-smi --query-gpu=memory.used --format=csv,noheader; ls -ls ~/lukehinojosa/$M; md5sum ~/lukehinojosa/bin/olmoe-g32"
        done
        md5sum $O
    } 2>&1 | tee "$out/versions.txt"
    ibv_devinfo -d rocep1s0f0 | grep -E "state|active_mtu"
    ibv_devinfo -d rocep1s0f1 | grep -E "state|active_mtu"
}

# The golden check: short prompts, the long prompt past DSA's budget, and 32 generated tokens.
golden()
{
    router golden 1500 --golden $G --generate 32 --max-step 4096 --payload f32
}

# The three-turn chat, recorded for the page.
chat()
{
    local secs
    secs=$(budget 1500)
    if [ "$secs" -lt 300 ]
    then
        echo "chat: skipped, $secs s left before $STOP_AT"
        return
    fi
    experts chat
    timeout --signal=TERM --kill-after=30 $secs $O rdma-router $M $PEERS --gpu --serve 127.0.0.1:8093 --ctx 16384 --record "$out/chat.jsonl" > "$out/chat.rank0-gx10-02.log" 2>&1 &
    local r0=$!
    for _ in $(seq 300)
    do
        if curl -s -o /dev/null http://127.0.0.1:8093/
        then
            break
        fi
        sleep 1
    done
    timeout $(( secs - 60 )) python3 src/engine/tools/ref/record_chat.py --base http://127.0.0.1:8093 --thinking --effort low --max-tokens 1200 | tee "$out/chat.txt"
    sleep 3
    kill -TERM $r0 2>/dev/null
    wait $r0
    finish chat
    gzip -f "$out/chat.jsonl"
}

long32()
{
    router 32k 1200 --tokens $D/repo.tokens --context 32640 --generate 128 --chunk 2048 --max-step 2048
}

long128()
{
    router 128k 2400 --tokens $D/repo.tokens --context 131040 --generate 32 --chunk 2048 --max-step 2048
}

for s in "$@"
do
    echo "== $s $(date +%T), stop at $STOP_AT"
    $s
done
echo G32_DONE $(date +%T)
