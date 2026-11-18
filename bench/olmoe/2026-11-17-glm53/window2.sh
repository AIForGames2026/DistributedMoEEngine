#!/bin/bash
# TASK-32 window on two Sparks: frees gx10-01 for GLM-5.3-Flash split between gx10-02 and gx10-01,
# then restores its lab server and checks it answers, whatever happens in between (trap on EXIT).
# gx10-03 is left alone. Run from Git Bash on the desktop: window2.sh SECTION... (sections are in
# run.sh on gx10-02, run with RANKS=2). Refuses to start if gx10-01's server is busy or stopped,
# or after DEADLINE; sections stop by STOP_AT, leaving the restore time to finish before 6 AM.
set -u
next()
{
    # The next time of day $1 comes, as a full date, so a window can start before or after midnight.
    local t
    t=$(date -d "$1" "+%F %H:%M")
    if [ "$(date -d "$t" +%s)" -le "$(date +%s)" ]
    then
        t=$(date -d "tomorrow $1" "+%F %H:%M")
    fi
    echo "$t"
}
DEADLINE="$(next "${DEADLINE:-05:00}")"
STOP_AT="$(next "${STOP_AT:-05:40}")"
OUT="$(cd "$(dirname "$0")" && pwd)/two"
mkdir -p "$OUT"
G2="ssh -o HostName=100.122.124.37 gx10-02"
# Runs $1 on admin@192.168.177.11 through gx10-02, quoted again so the second hop keeps it whole.
on1()
{
    $G2 "ssh -o ConnectTimeout=10 admin@192.168.177.11 $(printf '%q' "$1")"
}
LOG="$OUT/window.log"
P2=models/glm53-flash/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf
STOPPED1=0

say()
{
    echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"
}

busy()
{
    local h1
    h1=$($G2 "curl -s -m 10 http://100.68.96.118:8888/health")
    say "gx10-01 health: ${h1:0:120}"
    # A server its owner stopped means they are using the machine some other way.
    if ! on1 "docker ps -q -f name=^qwen38-27b-tf\$ -f status=running" | grep -q .
    then
        say "gx10-01's container is not running, so its owner may be using the machine"
        return 0
    fi
    [[ "$h1" != *'"busy": false'* || "$h1" != *'"requests_running": 0'* ]]
}

# Holds the watchdog's lock so its per-minute cron skips (it would restart the container), then stops the container.
free_spark()
{
    on1 "(nohup flock \$HOME/.glm-watchdog/lock sleep 10800 > /dev/null 2>&1 < /dev/null &)"
    on1 "docker stop -t 60 qwen38-27b-tf" && STOPPED1=1 && say "gx10-01: qwen38-27b-tf stopped, watchdog paused"
    on1 "free -g | head -2" | tee -a "$LOG"
}

# Starts the container again if this window stopped it, waits for it to answer, then gives the watchdog its lock back.
restore()
{
    say "restoring gx10-01's lab server"
    local ok1=1
    if [ $STOPPED1 = 1 ]
    then
        on1 "docker start qwen38-27b-tf"
        ok1=0
        for _ in $(seq 90)
        do
            if $G2 "curl -s -m 5 http://100.68.96.118:8888/health" | grep -q '"ok": true'
            then
                ok1=1
                say "gx10-01: qwen38-27b-tf healthy"
                break
            fi
            sleep 10
        done
    fi
    # Both flock and its sleep hold the lock; the anchors keep pkill from matching this shell's own command line.
    on1 "pkill -f '^flock .*glm-watchdog/lock sleep 10800'; pkill -f '^sleep 10800\$'; sleep 1; flock -n \$HOME/.glm-watchdog/lock true" && say "gx10-01: watchdog lock released" || say "WARNING: gx10-01's watchdog lock is still held"
    if [ $ok1 = 1 ]
    then
        say "gx10-01's lab server is back"
    else
        say "WARNING: gx10-01 not healthy after 15 minutes; check it by hand"
    fi
    for f in $($G2 "cd ~/lukehinojosa/bench-glm53/two && ls *.txt *.log *.jsonl.gz 2>/dev/null")
    do
        scp -q -o HostName=100.122.124.37 "gx10-02:lukehinojosa/bench-glm53/two/$f" "$OUT/"
    done
    say "logs copied to $OUT"
}

if [ "$(date -d "$DEADLINE" +%s)" -gt "$(date -d "$STOP_AT" +%s)" ]
then
    say "past the deadline (sections would stop at $STOP_AT), not starting"
    exit 1
fi
if busy
then
    say "gx10-01's lab server is busy, not starting"
    exit 1
fi
trap restore EXIT
trap 'exit 1' INT TERM HUP
say "window: deadline $DEADLINE, sections stop at $STOP_AT"
free_spark
# The watchdog would restart the container beside our ranks, so nothing loads unless its lock is really held.
if on1 "flock -n \$HOME/.glm-watchdog/lock true"
then
    say "gx10-01's watchdog lock is not held, stopping"
    exit 1
fi
say "gx10-01: watchdog lock confirmed held"
# gx10-01 holds experts 192 to 287 from the three-Spark split; rank 1 of 2 also needs 144 to 191.
say "== staging experts 144 to 191 on gx10-01"
$G2 "cd ~/lukehinojosa && tools/ref-venv5/bin/python bench-glm53/stage.py $P2 admin@192.168.177.11 --add 144 191" 2>&1 | tr '\r' '\n' | tail -2 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" != 0 ]
then
    say "staging failed, stopping"
    exit 1
fi
say "== rehearsal with the tiny model"
$G2 "RANKS=2 STOP_AT='$STOP_AT' O='${O:-}' EB='${EB:-}' M=models/glm53-tiny/tiny.gguf G=models/glm53-tiny/tiny.tfref bash ~/lukehinojosa/bench-glm53/run.sh golden" 2>&1 | tee -a "$LOG"
if ! $G2 "grep -q PASS ~/lukehinojosa/bench-glm53/two/golden.rank0-gx10-02.log"
then
    say "the tiny rehearsal did not pass, stopping"
    exit 1
fi
$G2 "cd ~/lukehinojosa/bench-glm53/two && for f in golden.*.log; do mv \$f rehearsal-tiny.\$f; done"
for s in "$@"
do
    say "== $s"
    $G2 "RANKS=2 STOP_AT='$STOP_AT' O='${O:-}' EB='${EB:-}' bash ~/lukehinojosa/bench-glm53/run.sh $s" 2>&1 | tee -a "$LOG"
done
