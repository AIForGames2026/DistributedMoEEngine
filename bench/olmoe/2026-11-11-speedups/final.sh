#!/bin/bash
# The final build (128-row tensor-core blocks by default, decode attention kernel): 16k checks and
# 128k all in RAM against the KV middle on NVMe.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
O=src/engine/zig-out-gen/bin/olmoe
$O local $M --golden $D/qwen3_forward.tfref --gpu > "$out/final-golden-plain.log" 2>&1
$O local $M --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 > "$out/final-golden-yarn.log" 2>&1
grep -h "PASS\|FAIL" "$out"/final-golden-*.log

A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
mkdir -p kvtmp
$O local $M $A --save /tmp/lh_s16.bin > "$out/final-16k.log" 2>&1
$O local $M $A --kv-offload kvtmp --compare /tmp/lh_s16.bin > "$out/final-16k-offload.log" 2>&1
TF_ATT_KERNEL=2 $O local $M $A --compare /tmp/lh_s16.bin > "$out/final-16k-drift.log" 2>&1
rm -f /tmp/lh_s16.bin
grep -hE "time|offload|PASS|FAIL" "$out"/final-16k*.log
echo SHORT_DONE

A="--tokens $D/repo.tokens --context 131040 --chunk 2048 --generate 32 --gpu --yarn 4"
/usr/bin/time -v $O local $M $A --save /tmp/lh_long_ram.bin > "$out/final-128k-ram.log" 2>&1
/usr/bin/time -v $O local $M $A --kv-offload kvtmp --compare /tmp/lh_long_ram.bin > "$out/final-128k-offload.log" 2>&1
rm -f /tmp/lh_long_ram.bin
rmdir kvtmp 2>/dev/null
grep -hE "context|time|offload|PASS|FAIL|Maximum resident" "$out"/final-128k-*.log
echo FINAL_DONE
