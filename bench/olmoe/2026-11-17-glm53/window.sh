#!/bin/bash
# TASK-32 window: frees gx10-01 and gx10-03 for a three-Spark GLM-5.3-Flash run, then restores
# both lab servers and checks they answer, whatever happens in between (trap on EXIT).
# Run from Git Bash on the desktop: window.sh SECTION... (the sections are in run.sh on gx10-02).
# Refuses to start if either server is busy, or after DEADLINE (local time, HH:MM). Sections
# stop by STOP_AT, leaving the restore (up to 15 minutes) time to finish before 6 AM.
set -u
DEADLINE="${DEADLINE:-05:00}"
STOP_AT="${STOP_AT:-05:40}"
OUT="$(cd "$(dirname "$0")" && pwd)"
G2="ssh -o HostName=100.122.124.37 gx10-02"
G1="$G2 ssh -o ConnectTimeout=10 admin@192.168.177.11"
G3="$G2 ssh -o ConnectTimeout=10 admin@192.168.197.13"
LOG="$OUT/window.log"

say()
{
    echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"
}

busy()
{
    local h1 r3
    h1=$($G2 "curl -s -m 10 http://100.68.96.118:8888/health")
    r3=$($G3 "curl -s -m 10 http://127.0.0.1:8001/metrics | grep '^vllm:num_requests_running' | awk '{s += \$2} END {print s + 0}'")
    say "gx10-01 health: ${h1:0:120}; gx10-03 requests running: $r3"
    [[ "$h1" != *'"busy": false'* || "$h1" != *'"requests_running": 0'* || "$r3" != "0" ]]
}

# gx10-01: hold the watchdog's lock so its per-minute cron skips (it would restart the
# container), then stop the container. gx10-03: stop its vLLM container.
free_sparks()
{
    $G1 "(nohup flock \$HOME/.glm-watchdog/lock sleep 10800 > /dev/null 2>&1 < /dev/null &)"
    $G1 "docker stop -t 60 qwen38-27b-tf" && say "gx10-01: qwen38-27b-tf stopped, watchdog paused"
    $G3 "docker stop -t 60 qwen38-flash" && say "gx10-03: qwen38-flash stopped"
    $G1 "free -g | head -2"
    $G3 "free -g | head -2"
}

# Starts both containers again, waits for each to answer, then gives gx10-01's watchdog its lock back.
restore()
{
    say "restoring the lab servers"
    $G3 "docker start qwen38-flash"
    $G1 "docker start qwen38-27b-tf"
    local ok1=0 ok3=0
    for _ in $(seq 90)
    do
        if [ $ok3 = 0 ] && [ "$($G3 "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health")" = 200 ]
        then
            ok3=1
            say "gx10-03: qwen38-flash healthy"
        fi
        if [ $ok1 = 0 ] && $G2 "curl -s -m 5 http://100.68.96.118:8888/health" | grep -q '"ok": true'
        then
            ok1=1
            say "gx10-01: qwen38-27b-tf healthy"
        fi
        if [ $ok1 = 1 ] && [ $ok3 = 1 ]
        then
            break
        fi
        sleep 10
    done
    # Both flock and its sleep hold the lock; the anchors keep pkill from matching this shell's own command line.
    $G1 "pkill -f '^flock .*glm-watchdog/lock sleep 10800'; pkill -f '^sleep 10800\$'; sleep 1; flock -n \$HOME/.glm-watchdog/lock true" && say "gx10-01: watchdog lock released" || say "WARNING: gx10-01's watchdog lock is still held"
    if [ $ok1 = 1 ] && [ $ok3 = 1 ]
    then
        say "both lab servers are back"
    else
        say "WARNING: gx10-01 healthy=$ok1, gx10-03 healthy=$ok3 after 15 minutes; check them by hand"
    fi
    for f in $($G2 "cd ~/lukehinojosa/bench-glm53 && ls versions.txt chat.txt chat.jsonl.gz golden.*.log chat.*.log 32k.*.log 128k.*.log 2>/dev/null")
    do
        scp -q -o HostName=100.122.124.37 "gx10-02:lukehinojosa/bench-glm53/$f" "$OUT/"
    done
    say "logs copied to $OUT"
}

if [[ "$(date +%H:%M)" > "$DEADLINE" ]]
then
    say "past $DEADLINE, not starting"
    exit 1
fi
if busy
then
    say "a lab server is busy, not starting"
    exit 1
fi
trap restore EXIT
trap 'exit 1' INT TERM HUP
free_sparks
for s in "$@"
do
    say "== $s"
    $G2 "STOP_AT=$STOP_AT bash ~/lukehinojosa/bench-glm53/run.sh $s" 2>&1 | tee -a "$LOG"
done
