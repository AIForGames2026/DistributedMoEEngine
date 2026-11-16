#!/bin/bash
# Tensor-core attention for 256-dim heads on gx10-02: kernel timings, Qwen3.6 against transformers, and long context.
# Run on gx10-02 from anywhere: run.sh SECTION... (attbench golden gpucheck long32 long128 long256); logs go next to this script.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O=src/engine/zig-out-gen/bin/olmoe
A=src/engine/zig-out-gen/bin/attbench
D=models/qwen36-35b-a3b
M=$D/Qwen3.6-35B-A3B-Q8_0.gguf
G=$D/qwen35_generate.tfref
L="--tokens $D/repo.tokens --chunk 2048 --gpu"
{
    echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
    echo "engine: $(strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ ]*tc256')"
    free -g | head -2
} > "$out/versions.txt"

# Both 256-dim kernels on random data: prefill chunks and decode, in each KV format.
attbench()
{
    {
        $A --hd 256 --heads 16 --kv-heads 2 --n 2048 --pos 0,16384,65536,129024,260096 --reps 3
        for f in bf16 fp8 bf16_fp8
        do
            $A --hd 256 --heads 16 --kv-heads 2 --n 2048 --pos 65536 --reps 3 --format $f
        done
        $A --hd 256 --heads 16 --kv-heads 2 --n 1 --pos 32000,129024,262000 --reps 5
        $A --heads 32 --kv-heads 4 --n 2048 --pos 16384 --reps 3
    } > "$out/attbench.log" 2>&1
}

golden()
{
    $O local $M --golden $G --generate 32 --gpu > "$out/golden-gpu.log" 2>&1
}

# The GPU's kernels against the CPU's, on a 2k prompt.
gpucheck()
{
    TF_GPU_ATTENTION=0 $O local $M $L --context 2048 --generate 32 --save /tmp/lh_tc256.bin > "$out/gpucheck-cpu.log" 2>&1
    $O local $M $L --context 2048 --generate 32 --compare /tmp/lh_tc256.bin > "$out/gpucheck-gpu.log" 2>&1
    rm -f /tmp/lh_tc256.bin
}

long32()
{
    /usr/bin/time -v $O local $M $L --context 32640 --generate 128 > "$out/32k.log" 2>&1
}

long128()
{
    /usr/bin/time -v $O local $M $L --context 131040 --generate 32 > "$out/128k.log" 2>&1
}

long256()
{
    /usr/bin/time -v $O local $M $L --context 262112 --generate 32 > "$out/256k.log" 2>&1
}

for s in "$@"
do
    echo "== $s $(date -u +%T)"
    $s
done
echo TC_DONE $(date -u +%T)
