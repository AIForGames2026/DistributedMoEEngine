#!/bin/bash
# From START (local HH:MM), waits until gx10-01's lab server has had no new requests for 15 minutes,
# then runs window2.sh with the given sections. Gives up at LAST so the window can finish in time.
set -u
START="${START:-}"
LAST="${LAST:-03:30}"
cd "$(dirname "$0")"
if [ -n "$START" ]
then
    wait_s=$(( $(date -d "$START" +%s) - $(date +%s) ))
    if [ $wait_s -gt 0 ]
    then
        echo "$(date +%T) waiting until $START"
        sleep $wait_s
    fi
fi
G2="ssh -o HostName=100.122.124.37 gx10-02"
quiet=0
prev=""
while true
do
    now=$(date +%H:%M)
    if [[ "$now" > "$LAST" && "$now" < "12:00" ]]
    then
        echo "$(date +%T) past $LAST, giving up"
        exit 1
    fi
    h=$($G2 "curl -s -m 10 http://100.68.96.118:8888/health")
    total=$(echo "$h" | grep -o '"requests_total": [0-9]*' | grep -o '[0-9]*$')
    # Rank 0 needs most of gx10-02's memory, so an engine already running there also counts as busy.
    engines=$($G2 "pgrep -c -f '^(src/engine/[^ ]*/bin/olmoe|/usr/bin/time -v src/engine)' || true")
    if [[ "$h" == *'"busy": false'* && "$h" == *'"requests_running": 0'* && -n "$total" && "$total" == "$prev" && "$engines" == "0" ]]
    then
        quiet=$(( quiet + 1 ))
    else
        quiet=0
    fi
    echo "$(date +%T) requests_total=$total engines on gx10-02=$engines quiet checks=$quiet"
    prev="$total"
    if [ $quiet -ge 3 ]
    then
        bash window2.sh "$@"
        exit $?
    fi
    sleep 300
done
