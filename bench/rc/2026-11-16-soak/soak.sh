#!/bin/bash
# TASK-15 soak: two engine processes on one Spark over RDMA, recording telemetry.
# Runs on the Spark: soak.sh MINUTES OUT_DIR BUILD
min=${1:-60}
out=${2:-/tmp/lh_rc}
build=${3:-unknown}
mkdir -p "$out"
cd ~/lukehinojosa || exit 1
source ./env.sh
M=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
G=models/olmoe_generate.tfref
bin=src/engine/zig-out-gen/bin/olmoe
echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version), engine $build" > "$out/versions.txt"
sample()
{
    while [ ! -f "$out/stop" ]
    do
        {
            echo "$(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) mem_used_mb $(free -m | awk 'NR==2{print $3}') avail_mb $(free -m | awk 'NR==2{print $7}') gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
            echo "  olmoe pid,rss_kb,etime: $(ps -o pid=,rss=,etime= -p "$(pgrep -d, -u "$USER" -x olmoe)" 2>/dev/null | tr -s ' ' | tr '\n' ';')"
            echo "  gpu_procs: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
            echo "  top cpu (user,pcpu,rss_kb,comm): $(ps -eo user=,pcpu=,rss=,comm= --sort=-pcpu | head -4 | tr -s ' ' | tr '\n' ';')"
        } >> "$out/load.txt"
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
/usr/bin/time -v $bin rdma $M --golden $G --payload f32 --gpu --generate 64 --minutes "$min" --record "$out/soak.jsonl" > "$out/soak.log" 2>&1
echo "rc=$?" >> "$out/soak.log"
touch "$out/stop"
wait $sp
echo SOAK_DONE >> "$out/soak.log"
