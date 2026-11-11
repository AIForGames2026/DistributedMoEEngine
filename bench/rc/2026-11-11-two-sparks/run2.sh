#!/bin/bash
# TASK-15b rerun: 40-minute two-node soak with per-round diagnostics and weight hashes.
out="$(dirname "$0")/rc2"
mkdir -p "$out"
B='source ~/lukehinojosa/env.sh; cd ~/lukehinojosa; /usr/bin/time -v src/engine/zig-out-gen/bin/olmoe'
M=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
G=models/olmoe_generate.tfref
C="--payload f32 --gpu --generate 64 --minutes 40"
sample() {
    while [ ! -f "$out/stop" ]; do
        for h in gx10-02 gx10-03; do
            ssh -o ConnectTimeout=20 $h "echo \"\$(hostname) \$(date -u +%FT%TZ) load \$(cut -d' ' -f1-3 /proc/loadavg) mem_used_mb \$(free -m | awk 'NR==2{print \$3}') avail_mb \$(free -m | awk 'NR==2{print \$7}') gpu \$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% gpu_procs \$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')\"" >> "$out/$1.load.txt" 2>/dev/null
        done
        for s in $(seq 6); do
            [ -f "$out/stop" ] && break
            sleep 10
        done
    done
}
for h in gx10-02 gx10-03; do
    ssh $h 'echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(source ~/lukehinojosa/env.sh; zig version)"' >> "$out/versions.txt"
done

rm -f "$out/stop"
sample twonode &
sp=$!
port=$(( 47980 + RANDOM % 15 ))
ssh gx10-03 "$B rdma-expert $M --port $port --dev rocep1s0f1 $C > /tmp/lh_rc_twonode.r1.log 2>&1; echo rc=\$? >> /tmp/lh_rc_twonode.r1.log" &
r1=$!
sleep 4
ssh gx10-02 "$B rdma-router $M --host 192.168.197.13 --port $port --dev rocep1s0f0 --golden $G $C > /tmp/lh_rc_twonode.r0.log 2>&1; echo rc=\$? >> /tmp/lh_rc_twonode.r0.log"
wait $r1
touch "$out/stop"
wait $sp
scp -q gx10-02:/tmp/lh_rc_twonode.r0.log "$out/twonode.rank0-gx10-02.log"
scp -q gx10-03:/tmp/lh_rc_twonode.r1.log "$out/twonode.rank1-gx10-03.log"
echo "TWONODE_DONE $(grep -h 'soak:' "$out"/twonode.rank0-gx10-02.log)"
echo RC2_DONE; grep -hE "soak round [0-9]+, prompt|soak:|weights in memory|rc=" "$out"/*.log
