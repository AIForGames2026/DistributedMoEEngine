#!/bin/bash
# TASK-31 on gx10-02: Qwen3.8-Flash-Next against transformers, split across two processes, and at long context,
# with experts streamed from NVMe through the expert cache.
# Run on gx10-02 from anywhere: run.sh SECTION... (golden split long32 long128); logs go next to this script.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O=src/engine/zig-out-q38k/bin/olmoe
D=models/qwen38-flash-next
M=$D/Qwen3.8-Flash-Next-Q8_0-00002-of-00002.gguf
G=$D/qwen38_golden.tfref
L="--tokens $D/repo.tokens --chunk 2048 --max-step 2048 --gpu --cache 12000"
{
    echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
    echo "engine: $(strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ "]*q38k')"
    echo "transformers: $(tools/ref-venv5/bin/python -c 'import transformers, torch; print(transformers.__version__, torch.__version__)')"
    sha256sum $D/Qwen3.8-Flash-Next-Q8_0-0000*-of-00002.gguf
    free -g | head -2
} > "$out/versions.txt"

# One process on the GPU, every expert through a 12,000-slot cache (63 GB) read from NVMe.
golden()
{
    $O local $M --golden $G --generate 32 --gpu --cache 12000 --max-step 4096 > "$out/golden-gpu.log" 2>&1
}

# Two processes over shared memory, experts split evenly, each with half the cache; activations in f32.
split()
{
    $O shm $M --golden $G --generate 32 --gpu --cache 6000 --max-step 4096 --payload f32 > "$out/split-shm.log" 2>&1
}

long32()
{
    /usr/bin/time -v $O local $M $L --context 32640 --generate 128 > "$out/32k.log" 2>&1
}

long128()
{
    /usr/bin/time -v $O local $M $L --context 131040 --generate 32 > "$out/128k.log" 2>&1
}

for s in "$@"
do
    echo "== $s $(date -u +%T)"
    $s
done
echo Q38_DONE $(date -u +%T)
