#!/bin/bash
# TASK-22: GPU router on gx10-03, CPU-only expert on the desktop (WSL2), over direct TCP.
# Run from Git Bash on the desktop. The Spark dials the desktop's campus address (TASK-21);
# fwd.py on Windows relays that connection into WSL2.
out="$(cd "$(dirname "$0")" && pwd)"
wout="/mnt/$(echo "${out:1:1}" | tr A-Z a-z)${out:2}"
DESK=184.171.155.166
M=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
G=models/olmoe_generate.tfref
C="--payload ${PAYLOAD:-f32} --generate 64"

ssh gx10-03 "echo \"\$(hostname): \$(uname -r), driver \$(nvidia-smi --query-gpu=driver_version --format=csv,noheader), \$(source ~/lukehinojosa/env.sh; zig version)\"" > "$out/versions.txt"
MSYS_NO_PATHCONV=1 wsl.exe bash -c "echo \"desktop WSL2: \$(uname -r), \$(lscpu | grep 'Model name' | sed 's/  */ /g'), \$(nproc) threads, \$(\$HOME/tools/zig/zig version)\"" >> "$out/versions.txt"

# run NAME BUILD [EXPERT ARGS]: one expert in WSL2, the router on gx10-03.
run() {
    local name=$1 build=$2
    shift 2
    local port=$(( 7500 + RANDOM % 50 ))
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "cd ~ && ~/tfs-scratch/$build/bin/olmoe expert ~/$M --port $port $C $* > '$wout/$name.rank1-desktop.log' 2>&1" &
    local ep=$!
    python "$out/fwd.py" $((port + 50)) $port $FWD_TO > "$out/$name.fwd.log" 2>&1 &
    local fp=$!
    sleep 3
    ssh gx10-03 "source ~/lukehinojosa/env.sh; cd ~/lukehinojosa; src/engine/zig-out-gen/bin/olmoe router $M --host $DESK --port $((port + 50)) --golden $G $C --gpu > /tmp/lh_t22.log 2>&1; echo rc=\$? >> /tmp/lh_t22.log"
    scp -q gx10-03:/tmp/lh_t22.log "$out/$name.rank0-gx10-03.log"
    ssh gx10-03 'rm -f /tmp/lh_t22.log'
    wait $ep
    kill $fp 2>/dev/null
    grep -hE "PASS|FAIL|tokens/s|expert FFNs|rc=" "$out/$name".rank*.log
}

echo "== cpu: expert built without -Dgpu"
run cpu t22-cpu
echo "== cpu2: repeat"
run cpu2 t22-cpu
echo "== gpu: same link, expert on the RTX 5070 Ti (comparison)"
run gpu t22-gpu --gpu
