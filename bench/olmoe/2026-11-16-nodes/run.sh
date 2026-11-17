#!/bin/bash
# TASK-33 on gx10-02: splits over 3 and 4 processes against one and two, goldens, and the tracker gathering 3 experts.
# Run on gx10-02 from anywhere: run.sh SECTION... (bits golden rdma tracker); logs go next to this script.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
B=src/engine/zig-out-gen/bin
O=$B/olmoe
OM=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
D=models/qwen3-30b-a3b
QM=$D/Qwen3-30B-A3B-Q8_0.gguf
{
    echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
    echo "engine: $(strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ ]*n33')"
    nproc
    free -g | head -2
} > "$out/versions.txt"

# Bit for bit, with every expert on the CPU: one process, then 2, 3 and 4, each compared with the first.
bits()
{
    L="--tokens $D/repo.tokens --context 512 --chunk 256 --generate 16 --payload f32 --max-step 256"
    $O local $QM $L --save /tmp/lh_n33.bin > "$out/bits-local.log" 2>&1
    for r in 2 3 4
    do
        $O shm $QM $L --ranks $r --compare /tmp/lh_n33.bin > "$out/bits-shm$r.log" 2>&1
    done
    rm -f /tmp/lh_n33.bin
}

# The TASK-10 and TASK-24 checks over 3 and 4 processes on shared memory.
golden()
{
    for r in 3 4
    do
        $O shm $OM --golden models/olmoe_generate.tfref --generate 32 --gpu --payload f32 --ranks $r > "$out/golden-olmoe-shm$r.log" 2>&1
        $O shm $QM --golden $D/qwen3_forward.tfref --gpu --payload f32 --ranks $r > "$out/golden-qwen3-shm$r.log" 2>&1
        $O shm $QM --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 --payload f32 --ranks $r > "$out/golden-qwen3-yarn-shm$r.log" 2>&1
    done
}

# 3 and 4 processes over RDMA on this machine's NIC.
rdma()
{
    for r in 3 4
    do
        $O rdma $OM --golden models/olmoe_generate.tfref --generate 32 --gpu --payload f32 --ranks $r > "$out/rdma-olmoe$r.log" 2>&1
    done
}

# expert NAME RANK PORT: starts an expert rank of 4 that joins the tracker.
expert()
{
    nohup sh -c "echo \$\$ > /tmp/lh_n33_$1.pid; exec $O expert $OM --port $3 --rank $2 --ranks 4 --tracker 127.0.0.1 --gpu --payload f32" > "$out/tracker-$1.log" 2>&1 < /dev/null &
}

# The router gathers 3 experts by itself; then a second run, where one is killed mid-generation.
tracker()
{
    nohup $B/tracker serve > "$out/tracker-serve.log" 2>&1 < /dev/null &
    local tk=$!
    sleep 1
    expert run-r1 1 48101
    expert run-r2 2 48102
    expert run-r3 3 48103
    $O router $OM --tracker 127.0.0.1 --ranks 4 --golden models/olmoe_generate.tfref --generate 32 --gpu --payload f32 > "$out/tracker-run-r0.log" 2>&1
    echo "rc=$?" >> "$out/tracker-run-r0.log"
    sleep 2
    expert kill-r1 1 48111
    expert kill-r2 2 48112
    expert kill-r3 3 48113
    $O router $OM --tracker 127.0.0.1 --ranks 4 --golden models/olmoe_generate.tfref --generate 64 --minutes 5 --gpu --payload f32 > "$out/tracker-kill-r0.log" 2>&1 &
    local rp=$!
    sleep 40
    echo "$(date -u +%H:%M:%S.%3N) SIGKILL expert rank 2" > "$out/tracker-signals.txt"
    kill -9 "$(cat /tmp/lh_n33_kill-r2.pid)"
    wait $rp
    echo "rc=$?" >> "$out/tracker-kill-r0.log"
    sleep 1
    for n in kill-r1 kill-r3
    do
        kill -9 "$(cat /tmp/lh_n33_$n.pid)" 2>/dev/null
    done
    kill $tk
    rm -f /tmp/lh_n33_*
}

for s in "$@"
do
    echo "== $s $(date -u +%T)"
    $s
done
echo N33_DONE $(date -u +%T)
