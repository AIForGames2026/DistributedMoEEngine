#!/bin/bash
# Diagnostic: dispatch --latency from gx10-03 to the desktop, through fwd.py into WSL2 and into native Windows.
out="$(cd "$(dirname "$0")" && pwd)"
DESK=184.171.155.166
R='source ~/lukehinojosa/env.sh; ~/lukehinojosa/src/engine/zig-out-gen/bin/dispatch router'
echo "== WSL2 through fwd.py"
MSYS_NO_PATHCONV=1 wsl.exe bash -c "~/tfs-scratch/tk/bin/dispatch expert --port 7601 --latency > /dev/null 2>&1" &
python "$out/fwd.py" 7651 7601 > /dev/null 2>&1 &
fp=$!
sleep 2
ssh gx10-03 "$R --host $DESK --port 7651 --latency" 2>&1 | grep -iE "latency|p50|PASS|FAIL"
wait %1 2>/dev/null; kill $fp
echo "== native Windows dispatch.exe"
"$out/../../../zig-out/bin/dispatch.exe" expert --port 7652 --latency > /dev/null 2>&1 &
sleep 2
ssh gx10-03 "$R --host $DESK --port 7652 --latency" 2>&1 | grep -iE "latency|p50|PASS|FAIL"
wait
