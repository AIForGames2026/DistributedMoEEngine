#!/bin/bash
# After run.sh: whether two f32 matmul kernels that only add in a different order (tiled, and the
# first Q8_0 kernel) also give different tokens at 16k.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
O=src/engine/zig-out-gen/bin/olmoe
A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
TF_MM_KERNEL=2 $O local $M $A --save /tmp/lh_t16.bin > "$out/drift-tiled-f32.log" 2>&1
TF_MM_KERNEL=1 $O local $M $A --compare /tmp/lh_t16.bin > "$out/drift-first-kernel.log" 2>&1
rm -f /tmp/lh_t16.bin
grep -hE "first token|PASS|FAIL" "$out"/drift-*.log
echo DRIFT_DONE
