#!/bin/bash
# TASK-15 cross-machine run: rank 0 in WSL2 on the desktop dials rank 1 on gx10-02 over Tailscale TCP.
# Runs from Git Bash on the desktop: cross.sh MINUTES OUT_DIR BUILD
min=${1:-10}
out=${2:-cross}
build=${3:-unknown}
mkdir -p "$out"
host=100.122.124.37
port=$(( 47980 + RANDOM % 15 ))
C="--payload f32 --gpu --generate 64 --minutes $min"
S='cd ~/lukehinojosa && source ./env.sh && src/engine/zig-out-gen/bin/olmoe'
W='cd ~ && ~/tf-out-gpu/bin/olmoe'
{
    ssh gx10-02 'echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(source ~/lukehinojosa/env.sh; zig version)"'
    wsl -e bash -lc 'echo "desktop WSL2: $(uname -r), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"'
    echo "engine $build"
} > "$out/versions.txt"
sample()
{
    while [ ! -f "$out/stop" ]
    do
        {
            ssh -o ConnectTimeout=20 gx10-02 'echo "$(hostname) $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg) avail_mb $(free -m | awk "NR==2{print \$7}") gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% olmoe_rss_kb $(ps -o rss= -p "$(pgrep -d, -u "$USER" -x olmoe)" 2>/dev/null | tr -s " \n" " ")"'
            wsl -e bash -lc 'echo "desktop $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg) avail_mb $(free -m | awk "NR==2{print \$7}") gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% olmoe_rss_kb $(ps -o rss= -C olmoe | tr -s " \n" " ")"'
            ping -n 3 $host | tail -1
        } >> "$out/load.txt" 2>/dev/null
        for s in $(seq 6)
        do
            [ -f "$out/stop" ] && break
            sleep 10
        done
    done
}
rm -f "$out/stop"
sample &
sp=$!
ssh gx10-02 "$S expert models/olmoe-1b-7b-0924-instruct-q8_0.gguf --port $port $C > /tmp/lh_cross.r1.log 2>&1; echo rc=\$? >> /tmp/lh_cross.r1.log" &
r1=$!
sleep 6
wsl -e bash -lc "$W router ~/models/olmoe-1b-7b-0924-instruct-q8_0.gguf --host $host --port $port --golden ~/models/olmoe_generate.tfref --record ~/lh_cross.jsonl $C > ~/lh_cross.r0.log 2>&1; echo rc=\$? >> ~/lh_cross.r0.log"
wait $r1
touch "$out/stop"
wait $sp
scp -q gx10-02:/tmp/lh_cross.r1.log "$out/cross.rank1-gx10-02.log"
wsl -e bash -lc "cat ~/lh_cross.r0.log" > "$out/cross.rank0-desktop.log"
wsl -e bash -lc "gzip -9 -c ~/lh_cross.jsonl" > "$out/cross.jsonl.gz"
echo CROSS_DONE
