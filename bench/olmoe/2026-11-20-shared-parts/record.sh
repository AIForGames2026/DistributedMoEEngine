#!/usr/bin/env bash
# TASK-42 on gx10-02: saves (or compares) every family's tokens and logits, bit for bit.
# Usage: record.sh TFSWARM save|compare OUT_DIR [model ...]   (tiny and the real models; all by default)
set -u
O=$1
MODE=$2
R=$3
W=${*:4}
W=${W:-tiny olmoe qwen3 qwen36 mixtral moonlight gptoss glm53}
M=~/lukehinojosa/models
B=~/lukehinojosa/bench-t42
mkdir -p "$R"
python3 -c "
import struct
open('$B/tiny.tokens', 'wb').write(struct.pack('<48I', *[(i * 37) % 200 + 1 for i in range(48)]))
open('$B/real.tokens', 'wb').write(struct.pack('<256I', *[(i * 7919) % 30000 + 100 for i in range(256)]))
"
run()
{
    local name=$1
    shift
    "$O" "$@" --$MODE "$B/base/$name.bin" > "$R/$name.log" 2>&1
    echo "$name: $(grep -E "PASS|FAIL|saved|error|recall" "$R/$name.log" | grep -v "^expert" | tail -n 2 | tr '\n' ' ')"
}
# A lookahead run must give the same bits as the run without it, so it compares with the GPU run.
ahead()
{
    local name=$1
    shift
    "$O" "$@" --compare "$B/base/$name-gpu.bin" > "$R/$name-ahead.log" 2>&1
    echo "$name-ahead: $(grep -E "PASS|FAIL|error|recall" "$R/$name-ahead.log" | grep -v "^expert" | tail -n 2 | tr '\n' ' ')"
}
mkdir -p "$B/base"
tiny()
{
    local s=$1
    local f=$B/tiny/${s}_tiny.gguf
    local a=(--context 40 --tokens "$B/tiny.tokens" --generate 16 --chunk 16)
    run tiny-$s-cpu local "$f" "${a[@]}"
    run tiny-$s-gpu local "$f" "${a[@]}" --gpu
    run tiny-$s-shm3 shm "$f" "${a[@]}" --ranks 3 --gpu --payload f32
    ahead tiny-$s local "$f" "${a[@]}" --gpu --lookahead 2 --extra 2
}
real()
{
    local name=$1 file=$2
    shift 2
    local a=(--context 256 --tokens "$B/real.tokens" --generate 24 --chunk 64 "$@")
    run $name-cpu local "$file" "${a[@]}"
    run $name-gpu local "$file" "${a[@]}" --gpu
    run $name-shm3 shm "$file" "${a[@]}" --ranks 3 --gpu --payload f32
    ahead $name local "$file" "${a[@]}" --gpu --lookahead 2 --extra 2
}
for w in $W
do
    case $w in
        tiny)
            for s in olmoe qwen3 qwen35 qwen38 glm53 mixtral deepseek gptoss
            do
                tiny $s
            done
            ;;
        olmoe) real olmoe $M/olmoe-1b-7b-0924-instruct-q8_0.gguf ;;
        qwen3) real qwen3 $M/qwen3-30b-a3b-gguf/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf ;;
        qwen36) real qwen36 $M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf ;;
        mixtral) real mixtral $M/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf ;;
        moonlight) real moonlight $M/moonlight-16b/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf ;;
        gptoss) real gptoss $M/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf ;;
        glm53)
            # 170 GB, so its experts stream from NVMe through the cache, as in TASK-32.
            f=$M/glm53-flash/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf
            a=(--context 256 --tokens "$B/real.tokens" --generate 24 --chunk 64 --gpu --cache 5000)
            run glm53-gpu local "$f" "${a[@]}"
            ahead glm53 local "$f" "${a[@]}" --lookahead 2 --extra 2
            ;;
    esac
done
