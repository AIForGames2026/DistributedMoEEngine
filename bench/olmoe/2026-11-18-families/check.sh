#!/usr/bin/env bash
# TASK-39 on gx10-02: the engine against transformers' goldens (refs.sh) and the tiny models.
# Usage: check.sh OLMOE OUT_DIR [model ...]   (models: tiny mixtral gptoss moonlight; all by default)
set -u
O=$1
R=$2
W=${*:3}
W=${W:-tiny mixtral gptoss moonlight}
M=~/lukehinojosa/models
mkdir -p "$R"
run()
{
    local name=$1
    shift
    "$O" "$@" > "$R/$name.log" 2>&1
    echo "$name: $(grep -E "PASS|FAIL|error" "$R/$name.log" | grep -v "^expert" | tail -n 2 | tr '\n' ' ')"
}
golden()
{
    local name=$1 file=$2 ref=$3
    run $name-local-gpu local "$file" --golden "$ref" --gpu --generate 32
    run $name-local-cpu local "$file" --golden "$ref" --generate 32
    # The split sends f32 activations, so it is held to the same bar as one process.
    run $name-shm3-gpu shm "$file" --golden "$ref" --ranks 3 --gpu --payload f32 --generate 32
}
for w in $W
do
    case $w in
        tiny)
            for s in mixtral deepseek gptoss
            do
                T=$M/tiny39
                run tiny-$s-local local $T/${s}_tiny.gguf --golden $T/${s}_tiny.tfref --payload f32 --generate 16
                run tiny-$s-shm3-gpu shm $T/${s}_tiny.gguf --golden $T/${s}_tiny.tfref --ranks 3 --gpu --payload f32 --generate 16
            done
            ;;
        mixtral) golden mixtral $M/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf $M/mixtral-8x7b/mixtral_golden.tfref ;;
        gptoss) golden gptoss $M/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf $M/gpt-oss-20b/gptoss_golden.tfref ;;
        moonlight) golden moonlight $M/moonlight-16b/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf $M/moonlight-16b/moonlight_golden.tfref ;;
    esac
done
