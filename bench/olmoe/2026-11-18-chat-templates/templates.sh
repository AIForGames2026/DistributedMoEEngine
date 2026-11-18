#!/usr/bin/env bash
# TASK-38 on gx10-02: the chat template files TASK-37's vocab.sh did not fetch, then gpt-oss's vocab GGUF again so it carries its template.
# Usage: templates.sh LLAMA_CPP_DIR
set -eu
L=$1
D=~/lukehinojosa/models/tok37
PY=~/lukehinojosa/tools/ref-venv/bin/python
get()
{
    mkdir -p $D/$1
    for f in "${@:3}"
    do
        curl -sfL -o $D/$1/$f https://huggingface.co/$2/resolve/main/$f
    done
}
# meta-llama's own repository is gated; unsloth's copy is not.
get llama4 unsloth/Llama-4-Scout-17B-16E-Instruct chat_template.jinja tokenizer_config.json tokenizer.json config.json special_tokens_map.json generation_config.json
# Kimi's tokenizer needs Moonshot's Python code, which is not run here: its prompts are compared as text only.
get kimi-k2 moonshotai/Kimi-K2-Instruct chat_template.jinja tokenizer_config.json config.json
get gpt-oss openai/gpt-oss-20b chat_template.jinja tokenizer_config.json
(cd $L && PYTHONPATH=gguf-py $PY convert_hf_to_gguf.py $D/gpt-oss --vocab-only --outfile $D/gpt-oss/gpt-oss-vocab.gguf > $D/gpt-oss/convert.log 2>&1)
