#!/bin/bash
# After run.sh, with the decode attention kernel and 128-row tensor-core blocks added: kernel
# benchmark, goldens, 16k checks, how far two f32 attention kernels drift apart at 16k (to set
# the tensor-core drift against), and 128k all in RAM for decode speed.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
B=src/engine/zig-out-gen/bin
O=$B/olmoe

echo "== attention kernels"
$B/attbench --pos 16384,129024 --reps 2 > "$out/attbench-final.log" 2>&1
$B/attbench --n 1 --pos 16384,131071 --reps 5 >> "$out/attbench-final.log" 2>&1
cat "$out/attbench-final.log"

echo "== golden checks"
$O local $M --golden $D/qwen3_forward.tfref --gpu > "$out/final-golden-plain.log" 2>&1
$O local $M --golden $D/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 > "$out/final-golden-yarn.log" 2>&1
grep -h "PASS\|FAIL" "$out"/final-golden-*.log

A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
echo "== 16k"
$O local $M $A --save /tmp/lh_s16.bin > "$out/final-16k.log" 2>&1
mkdir -p kvtmp
$O local $M $A --kv-offload kvtmp --compare /tmp/lh_s16.bin > "$out/final-16k-offload.log" 2>&1
echo "== 16k drift: wide f32 kernel throughout, then the first f32 kernel and the default kernels against it"
TF_ATT_KERNEL=2 $O local $M $A --save /tmp/lh_w16.bin > "$out/drift-wide.log" 2>&1
TF_ATT_KERNEL=1 $O local $M $A --compare /tmp/lh_w16.bin > "$out/drift-first.log" 2>&1
$O local $M $A --compare /tmp/lh_w16.bin > "$out/drift-default.log" 2>&1
rm -f /tmp/lh_s16.bin /tmp/lh_w16.bin
grep -hE "first token|offload|PASS|FAIL" "$out"/final-16k*.log "$out"/drift-*.log
echo SHORT_DONE

A="--tokens $D/repo.tokens --context 131040 --chunk 2048 --generate 32 --gpu --yarn 4"
echo "== 128k, all in RAM"
/usr/bin/time -v $O local $M $A > "$out/final-128k-ram.log" 2>&1
rmdir kvtmp 2>/dev/null
grep -hE "time|Maximum resident" "$out/final-128k-ram.log"
echo AFTER_DONE
