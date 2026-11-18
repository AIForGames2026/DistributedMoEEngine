#!/usr/bin/env bash
# TASK-36 on gx10-02: decode speed of Qwen3-30B-A3B as Q8_0, Q4_K_M and IQ4_XS, every expert on the GPU.
# Usage: speed.sh BUILD_DIR OUT_DIR   (each run twice; the second has the file in the page cache)
set -u
O=$1/bin/olmoe
R=$2
mkdir -p "$R"
M=~/lukehinojosa/models
Q=$M/qwen3-30b-a3b
G=$M/qwen3-30b-a3b-gguf
A="--tokens $Q/repo.tokens --context 1024 --generate 128 --gpu"
for m in q8_0:$Q/Qwen3-30B-A3B-Q8_0.gguf q4_k_m:$G/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf iq4_xs:$G/Qwen_Qwen3-30B-A3B-IQ4_XS.gguf
do
    name=${m%%:*}
    for i in 1 2
    do
        /usr/bin/time -v $O local ${m#*:} $A > "$R/$name-$i.log" 2>&1
        echo "EXIT $?" >> "$R/$name-$i.log"
    done
    echo "$name: $(grep -hiE 'tok/s|tokens/s|tokens per second' "$R/$name-2.log" | tail -n 3 | paste -sd ' ')"
done
