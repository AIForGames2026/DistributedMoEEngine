#!/bin/bash
# Reruns the cache runs that stopped with TooManyTokens (the expert cache needs --max-step), after run.sh finishes.
# Runs on gx10-02: fixup.sh  (waits for FINAL_DONE in out/progress.txt, then logs to out/fixup)
source ~/lukehinojosa/env.sh
L=~/lukehinojosa
W=$L/bench-final
O=$L/zig-out-final/bin/tfswarm
M=$L/models
QD=$M/qwen3-30b-a3b
Q8=$QD/Qwen3-30B-A3B-Q8_0.gguf
GD=$M/glm53-flash
GLM=$GD/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf
R=$W/out/fixup
while ! grep -q FINAL_DONE "$W/out/progress.txt"
do
    sleep 60
done
mkdir -p "$R"
run()
{
    local name=$1
    shift
    /usr/bin/time -v "$@" > "$R/$name.log" 2>&1
    echo "exit $?" >> "$R/$name.log"
    echo "$name: $(grep -hE "time to first token|PASS|FAIL|error|expert cache: [0-9]+ expert calls" "$R/$name.log" | grep -v "^expert" | cut -c1-220 | tail -n 3 | paste -sd ' ')" >> "$R/summary.txt"
}
echo "$(date -u +%FT%TZ) begin fixup" >> "$W/out/progress.txt"
run glm53-golden "$O" local $GLM --golden $GD/glm53_golden.tfref --gpu --cache 5000 --generate 16 --max-step 2048
Q="--tokens $QD/repo.tokens --context 512 --generate 128 --chunk 512 --gpu --max-step 2048"
for slots in 1536 3072
do
    run qwen3-cache$slots "$O" local $Q8 $Q --cache $slots
    run qwen3-cache$slots-ahead "$O" local $Q8 $Q --cache $slots --lookahead 2 --extra 2
done
echo "$(date -u +%FT%TZ) FIXUP_DONE" >> "$W/out/progress.txt"
