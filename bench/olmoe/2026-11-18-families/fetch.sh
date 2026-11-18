#!/usr/bin/env bash
# TASK-39 on gx10-02: the three models' GGUFs, plus the Hugging Face files their references and tokenizers need.
set -u
M=~/lukehinojosa/models
get()
{
    mkdir -p $M/$1
    for f in "${@:3}"
    do
        if [ ! -s $M/$1/$(basename $f) ]
        then
            curl -sfL --retry 5 -C - -o $M/$1/$(basename $f) https://huggingface.co/$2/resolve/main/$f || echo "FAILED $2/$f"
        fi
    done
}
get moonlight-16b moonshotai/Moonlight-16B-A3B-Instruct config.json tokenizer_config.json tiktoken.model tokenization_moonshot.py generation_config.json
get moonlight-16b gabriellarson/Moonlight-16B-A3B-Instruct-GGUF Moonlight-16B-A3B-Instruct-Q4_K_M.gguf
get gpt-oss-20b openai/gpt-oss-20b config.json tokenizer_config.json tokenizer.json special_tokens_map.json generation_config.json chat_template.jinja
get gpt-oss-20b ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf
cp -n $M/tok37/mixtral/{config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json} $M/mixtral-8x7b/ 2>/dev/null || { mkdir -p $M/mixtral-8x7b; cp -n $M/tok37/mixtral/{config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json} $M/mixtral-8x7b/; }
get mixtral-8x7b mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf
ls -l $M/moonlight-16b $M/gpt-oss-20b $M/mixtral-8x7b
