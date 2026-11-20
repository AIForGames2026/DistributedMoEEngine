#!/usr/bin/env bash
# TASK-42 on gx10-02: Qwen1.5-MoE-A2.7B-Chat and the tiny Qwen2-MoE against transformers' goldens.
# Usage: check.sh TFSWARM OUT_DIR
set -u
O=$1
R=$2
M=~/lukehinojosa/models/qwen2moe-a2.7b
T=~/lukehinojosa/bench-t42/tiny
mkdir -p "$R"
run()
{
    local name=$1
    shift
    "$O" "$@" > "$R/$name.log" 2>&1
    echo "$name: $(grep -E "PASS|FAIL|error" "$R/$name.log" | grep -v "^expert" | tail -n 2 | tr '\n' ' ')"
}
run tiny-qwen2moe-local local $T/qwen2moe_tiny.gguf --golden $T/qwen2moe_tiny.tfref --payload f32 --generate 16
run tiny-qwen2moe-local-gpu local $T/qwen2moe_tiny.gguf --golden $T/qwen2moe_tiny.tfref --payload f32 --generate 16 --gpu
run tiny-qwen2moe-shm3-gpu shm $T/qwen2moe_tiny.gguf --golden $T/qwen2moe_tiny.tfref --ranks 3 --gpu --payload f32 --generate 16
G=$M/Qwen1.5-MoE-A2.7B-Chat-Q8_0.gguf
run qwen2moe-local-gpu local $G --golden $M/qwen2moe_golden.tfref --gpu --generate 32
run qwen2moe-local-cpu local $G --golden $M/qwen2moe_golden.tfref --generate 32
# The split sends f32 activations, so it is held to the same bar as one process.
run qwen2moe-shm3-gpu shm $G --golden $M/qwen2moe_golden.tfref --ranks 3 --gpu --payload f32 --generate 32
run qwen2moe-shm3-gpu-fp8 shm $G --golden $M/qwen2moe_golden.tfref --ranks 3 --gpu --generate 32
run qwen2moe-local-gpu-ahead local $G --golden $M/qwen2moe_golden.tfref --gpu --generate 32 --lookahead 2 --extra 2
