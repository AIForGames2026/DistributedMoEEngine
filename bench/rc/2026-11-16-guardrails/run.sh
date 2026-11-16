#!/bin/bash
# TASK-17 runs, from Git Bash on the desktop. Rank 0 on gx10-02, rank 1 on gx10-03, RDMA over their cable.
out="$(dirname "$0")"
B='cd ~/lukehinojosa && source ./env.sh && src/engine/zig-out-gen/bin/olmoe'
OLD='cd ~/lukehinojosa && source ./env.sh && /tmp/lh_head/out/bin/olmoe'
M=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
Q=models/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf
T=/tmp/lh_olmoe_text.tokens
for h in gx10-02 gx10-03
do
    ssh $h 'echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(source ~/lukehinojosa/env.sh; zig version), load $(cut -d" " -f1-3 /proc/loadavg), avail $(free -m | awk "NR==2{print \$7}") MB, gpu procs: $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr "\n" ";")"'
done > "$out/versions.txt"

# Regression: the committed build against this one, both ranks forked on gx10-02, no new options.
R="--payload f32 --gpu --context 512 --chunk 64 --generate 256 --tokens $T"
ssh gx10-02 "$OLD rdma $M $R --save /tmp/lh_ref.bin" > "$out/regress-ref.log" 2>&1
for i in 1 2 3
do
    ssh gx10-02 "$OLD rdma $M $R --compare /tmp/lh_ref.bin" > "$out/regress-old-$i.log" 2>&1
    ssh gx10-02 "$B rdma $M $R --compare /tmp/lh_ref.bin" > "$out/regress-new-$i.log" 2>&1
done

# Two Sparks over RDMA, 4,032-token prompt in 64-token steps with a 3x spike in its middle third.
pair()
{
    local name=$1
    shift
    local port=$(( 47980 + RANDOM % 15 ))
    ssh gx10-03 "$B rdma-expert $M --port $port --dev rocep1s0f1 $* > /tmp/lh_g.r1.log 2>&1; echo rc=\$? >> /tmp/lh_g.r1.log" &
    local r1=$!
    sleep 4
    ssh gx10-02 "$B rdma-router $M --host 192.168.197.13 --port $port --dev rocep1s0f0 $* > /tmp/lh_g.r0.log 2>&1; echo rc=\$? >> /tmp/lh_g.r0.log"
    wait $r1
    scp -q gx10-02:/tmp/lh_g.r0.log "$out/$name.rank0-gx10-02.log"
    scp -q gx10-03:/tmp/lh_g.r1.log "$out/$name.rank1-gx10-03.log"
}
S="--payload f32 --gpu --context 4032 --chunk 64 --max-step 192 --spike 3 --generate 32 --tokens $T"
pair spike-unguarded $S
pair spike-guarded $S --capacity 1.25 --memory-budget 7000
pair refused $S --capacity 1.25 --memory-budget 4500
# Repeats, after spike-unguarded's 2.25 s wait: guarded-2, unguarded-2 and -3; then, with each
# phase's longest wait labelled by step, unguarded-4 to -7 and guarded-3 and -4.
pair spike-guarded-2 $S --capacity 1.25 --memory-budget 7000
for i in 2 3 4 5 6 7
do
    pair spike-unguarded-$i $S
done
for i in 3 4
do
    pair spike-guarded-$i $S --capacity 1.25 --memory-budget 7000
done

# Qwen3-30B-A3B in one process on gx10-02, the same spike, with a budget.
ssh gx10-02 "$B local $Q --gpu --context 4032 --chunk 64 --max-step 192 --spike 3 --generate 32 --tokens models/qwen3-30b-a3b/repo.tokens --capacity 1.25 --memory-budget 40000" > "$out/qwen3-spike-guarded.log" 2>&1
ssh gx10-02 'rm -f /tmp/lh_ref.bin /tmp/lh_g.r0.log'
ssh gx10-03 'rm -f /tmp/lh_g.r1.log'
echo GUARD_DONE
