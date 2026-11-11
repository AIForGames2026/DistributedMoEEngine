#!/bin/bash
# Prefill and decode speedups on gx10-02: kernel benchmark, Qwen3 goldens, 16k with the old and new
# kernels, then 128k all in RAM and with the KV middle on NVMe. Run from ~/lukehinojosa; logs go here.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
B=src/engine/zig-out-gen/bin
O=$B/olmoe
echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(zig version)" > "$out/versions.txt"

echo "== attention kernels"
$B/attbench --pos 0,16384,65536,129024 --reps 2 > "$out/attbench.log" 2>&1
$B/attbench --n 1 --pos 16384,131071 --reps 5 >> "$out/attbench.log" 2>&1
cat "$out/attbench.log"

echo "== golden checks, plain rope and YaRN 4"
$O local $M --golden $D/qwen3_forward.tfref --gpu > "$out/golden-plain.log" 2>&1
$O local $M --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 > "$out/golden-yarn.log" 2>&1
grep -h "PASS\|FAIL" "$out"/golden-*.log

A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
echo "== 16k, first matmul kernels and wide attention"
TF_MM_KERNEL=1 TF_ATT_KERNEL=2 $O local $M $A > "$out/16k-old-kernels.log" 2>&1
echo "== 16k, new kernels"
$O local $M $A --save /tmp/lh_s16.bin > "$out/16k-new.log" 2>&1
echo "== 16k, new kernels, offloaded"
mkdir -p kvtmp
$O local $M $A --kv-offload kvtmp --compare /tmp/lh_s16.bin > "$out/16k-new-offload.log" 2>&1
echo "== 16k, new matmul kernels with wide attention, against tensor cores"
TF_ATT_KERNEL=2 $O local $M $A --compare /tmp/lh_s16.bin > "$out/16k-wide-attention.log" 2>&1
rm -f /tmp/lh_s16.bin
grep -hE "time|offload|PASS|FAIL" "$out"/16k-*.log
echo SHORT_DONE

A="--tokens $D/repo.tokens --context 131040 --chunk 2048 --generate 32 --gpu --yarn 4"
echo "== 128k, all in RAM"
/usr/bin/time -v $O local $M $A --save /tmp/lh_long_ram.bin > "$out/128k-ram.log" 2>&1
echo "== 128k, middle offloaded to NVMe"
/usr/bin/time -v $O local $M $A --kv-offload kvtmp --compare /tmp/lh_long_ram.bin > "$out/128k-offload.log" 2>&1
rm -f /tmp/lh_long_ram.bin
rmdir kvtmp 2>/dev/null
grep -hE "context|time|offload|PASS|FAIL|Maximum resident" "$out"/128k-*.log
echo LONG_DONE
