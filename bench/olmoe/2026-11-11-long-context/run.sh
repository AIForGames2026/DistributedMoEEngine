#!/bin/bash
# TASK-12 on gx10-02: Qwen3-30B-A3B checks with YaRN, then 128k context all in RAM and with the KV middle on NVMe.
# Run on gx10-02 from ~/lukehinojosa; logs go next to this script.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
O=src/engine/zig-out-gen/bin/olmoe
echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(zig version), $(nvme list 2>/dev/null | grep -m1 nvme | awk '{print $3,$4,$5}')" > "$out/versions.txt"
free -g >> "$out/versions.txt"

echo "== golden checks, plain rope and YaRN 4"
$O local $M --golden $D/qwen3_forward.tfref --gpu > "$out/golden-plain.log" 2>&1
$O local $M --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 > "$out/golden-yarn.log" 2>&1
grep -h "PASS\|FAIL" "$out"/golden-*.log

A="--tokens $D/repo.tokens --context 131040 --chunk 2048 --generate 32 --gpu --yarn 4"
echo "== 128k, all in RAM"
/usr/bin/time -v $O local $M $A --save /tmp/lh_long_ram.bin > "$out/ram.log" 2>&1
echo "== 128k, middle offloaded to NVMe"
mkdir -p kvtmp
/usr/bin/time -v $O local $M $A --kv-offload kvtmp --compare /tmp/lh_long_ram.bin > "$out/offload.log" 2>&1
rm -f /tmp/lh_long_ram.bin
rmdir kvtmp 2>/dev/null
grep -hE "context|first token|offload|PASS|FAIL|Maximum resident|Elapsed" "$out"/ram.log "$out"/offload.log
echo LONG_DONE
