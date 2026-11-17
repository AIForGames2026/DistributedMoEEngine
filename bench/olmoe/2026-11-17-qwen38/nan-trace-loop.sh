#!/bin/bash
# Repeats the GPU golden check with TF_NAN_TRACE=1 until a run stops on a non-finite value.
source ~/lukehinojosa/env.sh
cd ~/lukehinojosa/models/qwen38-flash-next
B=~/lukehinojosa/src/engine/zig-out-q38/bin/olmoe
for i in $(seq 1 20)
do
    TF_NAN_TRACE=1 $B local Qwen3.8-Flash-Next-Q8_0-00002-of-00002.gguf --golden qwen38_golden.tfref --gpu --cache 12000 --max-step 4096 > nan_$i.log 2>&1
    echo "run $i: $(grep -c "^trace: layer [0-9]* expert" nan_$i.log) flagged calls, $(tail -1 nan_$i.log)" >> nanloop.log
    if grep -q NonFinite nan_$i.log
    then
        echo FOUND >> nanloop.log
        break
    fi
done
echo LOOP_DONE >> nanloop.log
