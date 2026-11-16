#!/bin/bash
# TASK-27 follow-up on gx10-02: keys in bf16 with values in fp8, against transformers, f32 and the other formats.
# Run on gx10-02 from anywhere: run.sh SECTION... (golden regress long32 prefix tier long128); logs go next to this script.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa || exit 1
O=src/engine/zig-out-gen/bin/olmoe
R=src/engine-ref/zig-out-ref/bin/olmoe
OM=models/olmoe-1b-7b-0924-instruct-q8_0.gguf
D=models/qwen3-30b-a3b
QM=$D/Qwen3-30B-A3B-Q8_0.gguf
Q="--tokens $D/repo.tokens --chunk 2048 --gpu --yarn 4"
{
    echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
    echo "new: $(strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ ]*kv27b')"
    echo "ref: $(strings $R | grep -m1 -o 'v0\.2\.0-alpha-[0-9]*-g[0-9a-f]*' )"
    free -g | head -2
} > "$out/versions.txt"

golden()
{
    $O local $OM --golden models/olmoe_generate.tfref --gpu --generate 64 --kv-format bf16_fp8 > "$out/olmoe-bf16_fp8.log" 2>&1
    $O local $QM --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 --kv-format bf16_fp8 > "$out/qwen3-bf16_fp8.log" 2>&1
}

# f32 against the committed build, bit for bit, at 32k.
regress()
{
    $R local $QM $Q --context 32640 --generate 128 --save /tmp/lh_kv_ref.bin > "$out/regress-ref.log" 2>&1
    $O local $QM $Q --context 32640 --generate 128 --kv-format f32 --compare /tmp/lh_kv_ref.bin > "$out/regress-new.log" 2>&1
    rm -f /tmp/lh_kv_ref.bin
}

# Every format at 32k against f32, twice each, alternating, for decode speed.
long32()
{
    for rep in 1 2
    do
        /usr/bin/time -v $O local $QM $Q --context 32640 --generate 128 --kv-format f32 --save /tmp/lh_kv32.bin > "$out/32k-f32-$rep.log" 2>&1
        for f in bf16 fp8 bf16_fp8
        do
            /usr/bin/time -v $O local $QM $Q --context 32640 --generate 128 --kv-format $f --compare /tmp/lh_kv32.bin > "$out/32k-$f-$rep.log" 2>&1
        done
    done
    rm -f /tmp/lh_kv32.bin
}

# Saved prefixes, with keys and values of different sizes, restore to exactly a cold run.
prefix()
{
    f=bf16_fp8
    P=~/lukehinojosa/prefixes_kv
    rm -rf $P && mkdir -p $P
    $O local $QM $Q --context 16384 --generate 8 --kv-format $f --save /tmp/lh_p16.bin > "$out/prefix-$f-16k-cold.log" 2>&1
    $O local $QM $Q --context 12000 --generate 8 --kv-format $f --prefix-cache $P > "$out/prefix-$f-12k-save.log" 2>&1
    $O local $QM $Q --context 16384 --generate 8 --kv-format $f --prefix-cache $P --compare /tmp/lh_p16.bin > "$out/prefix-$f-16k-partial.log" 2>&1
    $O local $QM $Q --context 16384 --generate 8 --kv-format $f --prefix-cache $P --compare /tmp/lh_p16.bin > "$out/prefix-$f-16k-full.log" 2>&1
    ls -l $P > "$out/prefix-$f-files.txt"
    rm -rf $P /tmp/lh_p16.bin
}

# The KV middle on NVMe gives exactly the all-in-memory run.
tier()
{
    f=bf16_fp8
    mkdir -p kvtmp_lh
    $O local $QM $Q --context 32640 --generate 32 --kv-format $f --save /tmp/lh_t.bin > "$out/tier-$f-ram.log" 2>&1
    $O local $QM $Q --context 32640 --generate 32 --kv-format $f --kv-offload kvtmp_lh --kv-window 8192 --compare /tmp/lh_t.bin > "$out/tier-$f-nvme.log" 2>&1
    rm -f /tmp/lh_t.bin
    rmdir kvtmp_lh
}

# 128k, all in memory.
long128()
{
    /usr/bin/time -v $O local $QM $Q --context 131040 --generate 32 --kv-format f32 --save /tmp/lh_kv128.bin > "$out/128k-f32.log" 2>&1
    /usr/bin/time -v $O local $QM $Q --context 131040 --generate 32 --kv-format bf16_fp8 --compare /tmp/lh_kv128.bin > "$out/128k-bf16_fp8.log" 2>&1
    rm -f /tmp/lh_kv128.bin
}

for s in "$@"
do
    echo "== $s $(date -u +%T)"
    $s
done
echo "KV_DONE $(date -u +%T)"
