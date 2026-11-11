#!/bin/bash
# After the 128k runs on gx10-02: the same 16k prefill with the build those runs used
# and with the later router and expert speedups, then offloaded against all in RAM on the new build.
out="$(cd "$(dirname "$0")" && pwd)"
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa
D=models/qwen3-30b-a3b
M=$D/Qwen3-30B-A3B-Q8_0.gguf
A="--tokens $D/repo.tokens --context 16384 --chunk 2048 --generate 8 --gpu --yarn 4"
echo "== 16k, build used for the 128k runs"
/tmp/lh_olmoe_old local $M $A > "$out/speedup-before.log" 2>&1
echo "== 16k, with the router and expert speedups"
src/engine/zig-out-gen/bin/olmoe local $M $A --save /tmp/lh_s16.bin > "$out/speedup-after.log" 2>&1
mkdir -p kvtmp
src/engine/zig-out-gen/bin/olmoe local $M $A --kv-offload kvtmp --compare /tmp/lh_s16.bin > "$out/speedup-after-offload.log" 2>&1
rmdir kvtmp
rm -f /tmp/lh_s16.bin /tmp/lh_olmoe_old
grep -hE "time|offload|PASS|FAIL" "$out"/speedup-*.log
echo SPEEDUP_DONE
