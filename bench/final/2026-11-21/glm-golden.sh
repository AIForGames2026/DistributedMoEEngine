#!/bin/bash
# GLM-5.3-Flash's golden check through the expert cache; its long prompt needs --max-step 4096, as in TASK-32.
# Runs on gx10-02: glm-golden.sh  (waits for FOLLOWUP_DONE, then logs to out/glm-golden)
source ~/lukehinojosa/env.sh
L=~/lukehinojosa
W=$L/bench-final
GD=$L/models/glm53-flash
R=$W/out/glm-golden
while ! grep -q FOLLOWUP_DONE "$W/out/progress.txt"
do
    sleep 60
done
mkdir -p "$R"
echo "$(date -u +%FT%TZ) begin glm-golden" >> "$W/out/progress.txt"
/usr/bin/time -v $L/zig-out-final/bin/tfswarm local $GD/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf --golden $GD/glm53_golden.tfref --gpu --cache 5000 --generate 32 --max-step 4096 > "$R/glm53-golden.log" 2>&1
echo "exit $?" >> "$R/glm53-golden.log"
echo "$(date -u +%FT%TZ) GLM_GOLDEN_DONE" >> "$W/out/progress.txt"
