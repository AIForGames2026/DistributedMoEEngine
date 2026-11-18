#!/usr/bin/env bash
# TASK-38 on gx10-02: every chat template rendered by the engine against transformers.
# Usage: check.sh TFTOK OUT_DIR   (run from the repo root; tokenizer files from bench/olmoe/2026-11-18-tokenizers/vocab.sh and ../templates.sh)
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
    $PY tools/ref/check_chat.py --tftok "$T" "$@" > "$R/$name.log" 2>&1
    echo "$name: $(tail -n 1 "$R/$name.log")"
}
run olmoe --hf-dir $V/olmoe --source $M/olmoe-1b-7b-0924-instruct-q8_0.gguf
run qwen3 --hf-dir $M/qwen3-30b-a3b --source $M/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf --gguf-template $M/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf
run qwen3-hf --hf-dir $M/qwen3-30b-a3b --source $M/qwen3-30b-a3b/tokenizer.json --config $M/qwen3-30b-a3b/tokenizer_config.json
run qwen36 --hf-dir $M/qwen36-35b-a3b --source $M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf --gguf-template $M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
run qwen38 --hf-dir $M/qwen38-flash-next --source $V/qwen38-vocab.gguf
run glm53 --hf-dir $M/glm53-flash --source $M/glm53-flash/GLM-5.3-Flash-Q4_K-00001-of-00002.gguf --gguf-template $M/glm53-flash/GLM-5.3-Flash-Q4_K-00001-of-00002.gguf
run mixtral --hf-dir $V/mixtral --source $V/mixtral/mixtral-vocab.gguf
run deepseek-v3 --hf-dir $V/deepseek-v3 --source $V/deepseek-v3/deepseek-v3-vocab.gguf
run gpt-oss --hf-dir $V/gpt-oss --source $V/gpt-oss/gpt-oss-vocab.gguf
run llama4 --hf-dir $V/llama4 --template-only
run kimi-k2 --hf-dir $V/kimi-k2 --template-only
