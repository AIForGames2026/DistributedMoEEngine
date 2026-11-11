#!/bin/bash
# TASK-28 on gx10-02: saved prompt prefixes against cold runs, Qwen3-30B-A3B with YaRN, at 16k and 128k.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
O=src/engine/zig-out-gen/bin/olmoe
P=~/lukehinojosa/prefixes
rm -rf $P && mkdir -p $P
echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), $(zig version)" > "$out/versions.txt"
df -h ~/lukehinojosa >> "$out/versions.txt"

A="--tokens $D/repo.tokens --chunk 2048 --generate 8 --gpu --yarn 4"
$O local $M $A --context 16384 --save /tmp/lh_p16.bin > "$out/16k-cold.log" 2>&1
$O local $M $A --context 12000 --prefix-cache $P --save /tmp/lh_p12.bin > "$out/12k-save.log" 2>&1
$O local $M $A --context 16384 --prefix-cache $P --compare /tmp/lh_p16.bin > "$out/16k-partial.log" 2>&1
$O local $M $A --context 16384 --prefix-cache $P --compare /tmp/lh_p16.bin > "$out/16k-full.log" 2>&1
$O local $M $A --context 12000 --prefix-cache $P --compare /tmp/lh_p12.bin > "$out/12k-full.log" 2>&1
TF_MM_KERNEL=2 $O local $M $A --context 16384 --prefix-cache $P > "$out/16k-other-kernel.log" 2>&1
ls -l $P > "$out/16k-files.txt"
rm -rf $P/* /tmp/lh_p16.bin /tmp/lh_p12.bin
for f in 16k-cold 12k-save 16k-partial 16k-full 12k-full 16k-other-kernel; do echo "== $f"; grep -hE "prefix cache|first token|PASS|FAIL|error" "$out/$f.log"; done
echo SHORT_DONE

A="--tokens $D/repo.tokens --context 131040 --chunk 2048 --generate 32 --gpu --yarn 4"
/usr/bin/time -v $O local $M $A --prefix-cache $P --save /tmp/lh_p128.bin > "$out/128k-cold.log" 2>&1
/usr/bin/time -v $O local $M $A --prefix-cache $P --compare /tmp/lh_p128.bin > "$out/128k-full.log" 2>&1
ls -l $P > "$out/128k-files.txt"
rm -rf $P /tmp/lh_p128.bin
for f in 128k-cold 128k-full; do echo "== $f"; grep -hE "prefix cache|first token|PASS|FAIL|error|Maximum resident" "$out/$f.log"; done
echo LONG_DONE
