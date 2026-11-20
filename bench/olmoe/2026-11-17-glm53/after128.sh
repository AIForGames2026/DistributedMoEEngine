#!/bin/bash
# Runs on gx10-02 while window2.sh waits on a paused 128k timeout: lets 128k finish, runs a full 32k,
# then resumes the timeout so the window restores gx10-01. Everything stops by HARD regardless.
set -u
T=$1 R=$2
HARD=06:30
cd ~/lukehinojosa/bench-glm53 || exit 1
resume()
{
    kill -CONT $T 2>/dev/null
    echo "$(date +%T) timeout resumed"
}
trap resume EXIT
past()
{
    [ "$(date +%s)" -ge "$(date -d "$1" +%s)" ]
}
while kill -0 $R 2>/dev/null
do
    if past $HARD
    then
        echo "$(date +%T) 128k still running at $HARD, stopping it"
        kill -TERM $R
        sleep 30
        kill -9 $R 2>/dev/null
        break
    fi
    sleep 10
done
echo "$(date +%T) 128k router exited"
# The 128k expert rank on gx10-01 must be gone before another loads beside it.
ssh -n admin@192.168.177.11 'kill $(cat /tmp/lh_g32.pid) 2>/dev/null; sleep 3; kill -9 $(cat /tmp/lh_g32.pid) 2>/dev/null; rm -f /tmp/lh_g32.pid; cp /tmp/lh_g32_128k.log /tmp/lh_g32_128k.keep 2>/dev/null; free -g | sed -n 2p'
# A full 32k needs about 27 minutes; it gets whatever is left before 07:35.
if past 07:00
then
    echo "$(date +%T) too late for a full 32k, skipping it"
    exit 0
fi
for f in two/32k.rank0-gx10-02.log two/32k.rank1-gx10-01.log
do
    [ -f "$f" ] && mv "$f" "${f/32k./32k-capped.}"
done
sed 's/router 32k 1200/router 32k 2400/' run.sh > run32.sh
RANKS=2 STOP_AT=07:35 O=src/engine/zig-out-g32m/bin/olmoe EB=bin/olmoe-g32m bash run32.sh long32
echo "$(date +%T) AFTER128_DONE"
