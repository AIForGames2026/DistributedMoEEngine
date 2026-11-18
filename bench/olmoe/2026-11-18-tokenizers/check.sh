#!/usr/bin/env bash
# TASK-37 on gx10-02: every model's tokenizer against Hugging Face, from its GGUF.
# Usage: check.sh TFTOK OUT_DIR   (run from the repo root, after vocab.sh)
set -u
T=$1
R=$2
mkdir -p "$R"
M=~/lukehinojosa/models
V=$M/tok37
PY=~/lukehinojosa/tools/ref-venv/bin/python
run()
{
    local name=$1
    shift
    $PY tools/ref/check_tokenizer.py --tftok "$T" --random 5000 --long src/tokenizer.zig --long Plan.md --long README.md "$@" > "$R/$name.log" 2>&1
    echo "$name: $(tail -n 1 "$R/$name.log")"
}
run mixtral --tokenizer $V/mixtral/tokenizer.json --source $V/mixtral/mixtral-vocab.gguf
run deepseek-v3 --tokenizer $V/deepseek-v3/tokenizer.json --source $V/deepseek-v3/deepseek-v3-vocab.gguf
run gpt-oss --tokenizer $V/gpt-oss/tokenizer.json --source $V/gpt-oss/gpt-oss-vocab.gguf
run olmoe --tokenizer $V/olmoe/tokenizer.json --source $M/olmoe-1b-7b-0924-instruct-q8_0.gguf
run qwen3 --tokenizer $M/qwen3-30b-a3b/tokenizer.json --source $M/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf
run qwen3-q4_k_m --tokenizer $M/qwen3-30b-a3b/tokenizer.json --source $M/qwen3-30b-a3b-gguf/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf
run qwen3-iq4_xs --tokenizer $M/qwen3-30b-a3b/tokenizer.json --source $M/qwen3-30b-a3b-gguf/Qwen_Qwen3-30B-A3B-IQ4_XS.gguf
run qwen36 --tokenizer $M/qwen36-35b-a3b/tokenizer.json --source $M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
run qwen38 --tokenizer $M/qwen38-flash-next/tokenizer.json --source $V/qwen38-vocab.gguf
run glm53 --tokenizer $M/glm53-flash/tokenizer.json --source $M/glm53-flash/GLM-5.3-Flash-Q4_K-00001-of-00002.gguf
for v in llama-bpe bert-bge
do
    echo "refuse $v: $("$T" ~/lukehinojosa/tools/llama.cpp/models/ggml-vocab-$v.gguf --info 2>&1 | head -n 1)"
done
