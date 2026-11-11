#!/bin/bash
# TASK-09: swarm tracker on gx10-02, engine ranks on gx10-02 (router) and gx10-03 (expert) over RDMA.
out="$(dirname "$0")"
E='source ~/lukehinojosa/env.sh; cd ~/lukehinojosa'
BIN=src/engine/zig-out-gen/bin
M=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
G=models/olmoe_generate.tfref
TK=192.168.197.12
C="--payload f32 --gpu"
# signal SIG NAME: signals the expert on gx10-03, logging the time on gx10-03 just before.
signal() {
    ssh gx10-03 "echo \"\$(date -u +%H:%M:%S.%3N) SIG$1 expert ($2) on gx10-03\"; kill -$1 \$(cat /tmp/lh_tk_$2.pid)" | tee -a "$out/signals.txt"
}

for h in gx10-02 gx10-03; do
    ssh $h "echo \"\$(hostname): \$(uname -r), driver \$(nvidia-smi --query-gpu=driver_version --format=csv,noheader), \$($E; zig version), \$(timedatectl timesync-status | grep Offset | xargs)\"" >> "$out/versions.txt"
done

ssh gx10-02 "$E; nohup $BIN/tracker serve > /tmp/lh_tk_tracker.log 2>&1 < /dev/null & echo \$! > /tmp/lh_tk_tracker.pid"
sleep 1
ssh gx10-03 "$E; nohup $BIN/tracker watch $TK > /tmp/lh_tk_watch.log 2>&1 < /dev/null & echo \$! > /tmp/lh_tk_watch.pid"

echo "== bench: 16 peers on gx10-03 over the direct cable, then 16 on gx10-02"
ssh gx10-03 "$E; $BIN/tracker bench $TK --peers 16 --rounds 20" > "$out/bench-from-gx10-03.txt" 2>&1
ssh gx10-02 "$E; $BIN/tracker bench 127.0.0.1 --peers 16 --rounds 20" > "$out/bench-from-gx10-02.txt" 2>&1
cat "$out"/bench-from-*.txt

# expert NAME PORT: starts rank 1 on gx10-03 and records its pid.
expert() {
    ssh gx10-03 "$E; nohup sh -c 'echo \$\$ > /tmp/lh_tk_$1.pid; exec $BIN/olmoe rdma-expert $M --port $2 --dev rocep1s0f1 --tracker $TK --advertise 192.168.197.13 $C' > /tmp/lh_tk_$1.r1.log 2>&1 < /dev/null &"
}
# router NAME ARGS...: runs rank 0 on gx10-02 in the background.
router() {
    local name=$1
    shift
    ssh gx10-02 "$E; $BIN/olmoe rdma-router $M --tracker 127.0.0.1 --dev rocep1s0f0 --golden $G $C $* > /tmp/lh_tk_$name.r0.log 2>&1; echo rc=\$? >> /tmp/lh_tk_$name.r0.log" &
}

for sig in KILL STOP; do
    name=$(echo $sig | tr A-Z a-z)
    echo "== $name: router waits, expert joins, SIG$sig to the expert after 60 s of generation"
    router $name --generate 64 --minutes 10
    rp=$!
    sleep 3
    expert $name $(( 47980 + RANDOM % 15 ))
    sleep 60
    signal $sig $name
    wait $rp
    ssh gx10-03 "kill -9 \$(cat /tmp/lh_tk_$name.pid) 2>/dev/null"
    sleep 2
done

echo "== resume: expert stopped before the router starts, resumed 5 s later"
expert resume $(( 47980 + RANDOM % 15 ))
sleep 25
signal STOP resume
sleep 2
router resume
rp=$!
sleep 5
signal CONT resume
wait $rp
sleep 2

ssh gx10-03 'kill $(cat /tmp/lh_tk_watch.pid)'
ssh gx10-02 'kill $(cat /tmp/lh_tk_tracker.pid)'
for f in tracker kill.r0 stop.r0 resume.r0; do
    scp -q gx10-02:/tmp/lh_tk_$f.log "$out/$f-gx10-02.log"
done
for f in watch kill.r1 stop.r1 resume.r1; do
    scp -q gx10-03:/tmp/lh_tk_$f.log "$out/$f-gx10-03.log"
done
ssh gx10-02 'rm -f /tmp/lh_tk_*'
ssh gx10-03 'rm -f /tmp/lh_tk_*'
echo TRACKER_DONE
grep -hE "refusing|PASS|FAIL|rc=|lists expert|joined the tracker" "$out"/*.log
