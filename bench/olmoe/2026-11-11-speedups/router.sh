#!/bin/bash
# The final build plus the decode router back on the CPU: goldens and the 16k checks again.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
O=src/engine/zig-out-gen/bin/olmoe
$O local $M --golden $D/qwen3_forward.tfref --gpu > "$out/router-golden-plain.log" 2>&1
$O local $M --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 > "$out/router-golden-yarn.log" 2>&1
A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
mkdir -p kvtmp
$O local $M $A --save /tmp/lh_s16.bin > "$out/router-16k.log" 2>&1
$O local $M $A --kv-offload kvtmp --compare /tmp/lh_s16.bin > "$out/router-16k-offload.log" 2>&1
rm -f /tmp/lh_s16.bin
rmdir kvtmp
grep -hE "time|offload|PASS|FAIL" "$out"/router-*.log
echo ROUTER_DONE
