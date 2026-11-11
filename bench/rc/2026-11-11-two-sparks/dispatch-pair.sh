#!/bin/bash
# usage: d2.sh RUNS dispatch-options...  (rank 1 on gx10-03, rank 0 on gx10-02, RDMA over the direct cable)
runs=$1
shift
D='cd ~/lukehinojosa/src/engine; zig-out-x/bin/dispatch'
pass=0
for i in $(seq 1 $runs); do
    port=$(( 47930 + RANDOM % 50 ))
    ssh gx10-03 "$D rdma-expert --port $port --dev rocep1s0f1 $* > /tmp/lh_d2.r1.log 2>&1; echo rc=\$? >> /tmp/lh_d2.r1.log" &
    r1=$!
    sleep 2
    out=$(ssh gx10-02 "$D rdma-router --host 192.168.197.13 --port $port --dev rocep1s0f0 $* 2>&1; echo rc=\$?")
    wait $r1
    r1out=$(ssh gx10-03 "cat /tmp/lh_d2.r1.log")
    if echo "$out" | grep -q "rc=0" && echo "$r1out" | grep -q "rc=0"; then
        pass=$((pass + 1))
    else
        echo "--- run $i failed"
        echo "$out"
        echo "$r1out"
    fi
    if [ $i -eq 1 ] || [ $runs -le 5 ]; then
        echo "--- run $i rank 0:"
        echo "$out" | tail -6
        echo "--- run $i rank 1:"
        echo "$r1out" | tail -3
    fi
done
echo "PASSED $pass of $runs: $*"
