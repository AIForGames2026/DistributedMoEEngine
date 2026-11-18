#!/usr/bin/env bash
# TASK-37 on gx10-02: Hugging Face tokenizer files, and vocab-only GGUFs made from them by llama.cpp's converter.
# Usage: vocab.sh LLAMA_CPP_DIR   (llama.cpp at ffc87c3a; the venv needs transformers and sentencepiece)
set -eu
L=$1
D=~/lukehinojosa/models/tok37
PY=~/lukehinojosa/tools/ref-venv/bin/python
for p in mixtral:mistralai/Mixtral-8x7B-Instruct-v0.1 deepseek-v3:deepseek-ai/DeepSeek-V3 gpt-oss:openai/gpt-oss-20b olmoe:allenai/OLMoE-1B-7B-0924-Instruct
do
    n=${p%%:*}
    r=${p#*:}
    mkdir -p $D/$n
    for f in config.json tokenizer.json tokenizer_config.json tokenizer.model special_tokens_map.json generation_config.json
    do
        curl -sfL -o $D/$n/$f https://huggingface.co/$r/resolve/main/$f || rm -f $D/$n/$f
    done
done
for n in mixtral deepseek-v3 gpt-oss
do
    (cd $L && PYTHONPATH=gguf-py $PY convert_hf_to_gguf.py $D/$n --vocab-only --outfile $D/$n/$n-vocab.gguf > $D/$n/convert.log 2>&1)
done
# Qwen3.8's converter needs the checkpoint even for the vocabulary; its GGUF was deleted after TASK-31.
S=$(ls -d ~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/*/)
(cd $L && PYTHONPATH=gguf-py $PY convert_hf_to_gguf.py $S --vocab-only --outfile $D/qwen38-vocab.gguf > $D/qwen38-convert.log 2>&1)
