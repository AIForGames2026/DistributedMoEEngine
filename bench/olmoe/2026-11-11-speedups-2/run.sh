#!/bin/bash
# Tensor-core Q8_0 matmuls and one wait per small GPU call, on gx10-02: matmul and memory benchmarks,
# goldens, 16k against the tiled f32 matmul, then 128k all in RAM against the KV middle on NVMe.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
B=src/engine/zig-out-gen/bin
O=$B/olmoe
echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(zig version)" > "$out/versions.txt"
for a in map host; do
  $B/attbench --mm --reps 3 --alloc $a
  $B/attbench --n 1 --pos 131071 --reps 5 --alloc $a
done > "$out/attbench.log" 2>&1
cat "$out/attbench.log"

$O local $M --golden $D/qwen3_forward.tfref --gpu > "$out/golden-plain.log" 2>&1
$O local $M --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 > "$out/golden-yarn.log" 2>&1
grep -h "PASS\|FAIL" "$out"/golden-*.log

A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
mkdir -p kvtmp
$O local $M $A --save /tmp/lh_s16.bin > "$out/16k.log" 2>&1
$O local $M $A --kv-offload kvtmp --compare /tmp/lh_s16.bin > "$out/16k-offload.log" 2>&1
TF_MM_KERNEL=2 $O local $M $A --compare /tmp/lh_s16.bin > "$out/16k-tiled-f32.log" 2>&1
rm -f /tmp/lh_s16.bin
grep -hE "time|offload|PASS|FAIL" "$out"/16k*.log
echo SHORT_DONE

A="--tokens $D/repo.tokens --context 131040 --chunk 2048 --generate 32 --gpu --yarn 4"
/usr/bin/time -v $O local $M $A --save /tmp/lh_long_ram.bin > "$out/128k-ram.log" 2>&1
/usr/bin/time -v $O local $M $A --kv-offload kvtmp --compare /tmp/lh_long_ram.bin > "$out/128k-offload.log" 2>&1
rm -f /tmp/lh_long_ram.bin
rmdir kvtmp 2>/dev/null
grep -hE "context|time|offload|PASS|FAIL|Maximum resident" "$out"/128k-*.log
echo LONG_DONE
