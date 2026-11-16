#!/bin/bash
# TASK-29 on gx10-02: Qwen3.6-35B-A3B (Qwen3.5-MoE) against transformers, split across processes, and at long context.
# Run on gx10-02 from anywhere: run.sh SECTION... (golden split gpucheck long32 long128 long256); logs go next to this script.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O=src/engine/zig-out-gen/bin/olmoe
D=models/qwen36-35b-a3b
M=$D/Qwen3.6-35B-A3B-Q8_0.gguf
G=$D/qwen35_generate.tfref
L="--tokens $D/repo.tokens --chunk 2048 --gpu"
{
    echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
    echo "engine: $(strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ ]*q35')"
    echo "transformers: $(tools/ref-venv5/bin/python -c 'import transformers, torch; print(transformers.__version__, torch.__version__)')"
    sha256sum $M | cut -c1-64
    free -g | head -2
} > "$out/versions.txt"

# One process, on the CPU and on the GPU.
golden()
{
    $O local $M --golden $G --generate 32 > "$out/golden-cpu.log" 2>&1
    $O local $M --golden $G --generate 32 --gpu > "$out/golden-gpu.log" 2>&1
}

# Two processes, experts split between them, activations sent in f32: shared memory, then RDMA through the NIC.
split()
{
    $O shm $M --golden $G --generate 32 --gpu --payload f32 > "$out/split-shm.log" 2>&1
    $O rdma $M --golden $G --generate 32 --gpu --payload f32 > "$out/split-rdma.log" 2>&1
}

# The GPU's attention and delta-rule kernels against the CPU's, on a 2k prompt.
gpucheck()
{
    TF_GPU_ATTENTION=0 $O local $M $L --context 2048 --generate 32 --save /tmp/lh_q35c.bin > "$out/gpucheck-cpu.log" 2>&1
    $O local $M $L --context 2048 --generate 32 --compare /tmp/lh_q35c.bin > "$out/gpucheck-gpu.log" 2>&1
    rm -f /tmp/lh_q35c.bin
}

long32()
{
    /usr/bin/time -v $O local $M $L --context 32640 --generate 128 > "$out/32k.log" 2>&1
}

long128()
{
    /usr/bin/time -v $O local $M $L --context 131040 --generate 32 > "$out/128k.log" 2>&1
}

# The model's whole native context.
long256()
{
    /usr/bin/time -v $O local $M $L --context 262112 --generate 32 > "$out/256k.log" 2>&1
}

for s in "$@"
do
    echo "== $s $(date -u +%T)"
    $s
done
echo Q35_DONE $(date -u +%T)
