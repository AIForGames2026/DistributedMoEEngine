#!/usr/bin/env bash
# TASK-36 on gx10-02: every existing golden with the new build, then the two new GGUFs.
# Usage: goldens.sh BUILD_DIR OUT_DIR [which...]   (which: old new; default both)
set -u
O=$1/bin/olmoe
R=$2
mkdir -p "$R"
M=~/lukehinojosa/models
Q=$M/qwen3-30b-a3b
G=$M/qwen3-30b-a3b-gguf
Q6=$M/qwen36-35b-a3b
L=$M/glm53-flash
which=${*:3}
which=${which:-old new}
run()
{
    local name=$1
    shift
    "$@" > "$R/$name.log" 2>&1
    echo "EXIT $?" >> "$R/$name.log"
    echo "$name: $(grep -hE 'PASS|FAIL' "$R/$name.log" | tail -n 2 | paste -sd ' ') $(tail -n 1 "$R/$name.log")"
}
if [[ $which == *old* ]]
then
    run olmoe-local $O local $M/olmoe-1b-7b-0924-instruct-q8_0.gguf --golden $M/olmoe_generate.tfref --generate 16 --gpu
    run olmoe-shm $O shm $M/olmoe-1b-7b-0924-instruct-q8_0.gguf --golden $M/olmoe_generate.tfref --generate 16 --gpu --payload f32
    run qwen3-local $O local $Q/Qwen3-30B-A3B-Q8_0.gguf --golden $Q/qwen3_forward.tfref --gpu
    run qwen3-cache $O local $Q/Qwen3-30B-A3B-Q8_0.gguf --golden $Q/qwen3_forward.tfref --gpu --cache 3000 --max-step 2048
    run qwen3-yarn $O local $Q/Qwen3-30B-A3B-Q8_0.gguf --golden $Q/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32
    run qwen36 $O local $Q6/Qwen3.6-35B-A3B-Q8_0.gguf --golden $Q6/qwen35_generate.tfref --generate 32 --gpu
    run glm53 $O local $L/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf --golden $L/glm53_golden.tfref --generate 32 --gpu --cache 5000 --max-step 4096
fi
if [[ $which == *new* ]]
then
    for q in q4_k_m iq4_xs
    do
        f=$G/Qwen_Qwen3-30B-A3B-$(echo $q | tr a-z A-Z).gguf
        run qwen3-$q $O local $f --golden $G/qwen3_${q}_generate.tfref --generate 32 --gpu
        run qwen3-$q-cpu $O local $f --golden $G/qwen3_${q}_generate.tfref --generate 32
        run qwen3-$q-shm $O shm $f --golden $G/qwen3_${q}_generate.tfref --generate 32 --gpu --payload f32
        run qwen3-$q-cache $O local $f --golden $G/qwen3_${q}_generate.tfref --generate 32 --gpu --cache 3000 --max-step 2048
    done
fi
