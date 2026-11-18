#!/usr/bin/env bash
# TASK-39 on gx10-02: transformers' goldens for the three real models, in float32 on the GGUFs' weights.
# Mixtral runs beside gpt-oss then Moonlight, 10 threads each; each writes REF_<name>.done when finished.
# Usage: refs.sh ENGINE_DIR OUT_DIR [mixtral] [others]  (both by default)
set -u
E=$1
O=$2
W=${*:3}
W=${W:-mixtral others}
M=~/lukehinojosa/models
PY=~/lukehinojosa/tools/ref-venv/bin/python
# Moonshot's tokenizer code is cached under our own folder, not the shared ~/.cache.
export HF_HOME=~/lukehinojosa/hf-home
cd $E
if [[ " $W " == *" mixtral "* ]]
then
(
    $PY tools/ref/ref_mixtral.py --gguf $M/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf --config $M/mixtral-8x7b/config.json \
        --tokenizer $M/mixtral-8x7b/tokenizer.json --out $M/mixtral-8x7b/mixtral_golden.tfref --generate 32 --threads 10 > $O/ref_mixtral.log 2>&1
    echo "exit $?" > $O/REF_mixtral.done
) &
fi
if [[ " $W " == *" others "* ]]
then
(
    $PY tools/ref/ref_gptoss.py --gguf $M/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf --config $M/gpt-oss-20b/config.json \
        --tokenizer $M/gpt-oss-20b/tokenizer.json --out $M/gpt-oss-20b/gptoss_golden.tfref --generate 32 --threads 10 > $O/ref_gptoss.log 2>&1
    echo "exit $?" > $O/REF_gptoss.done
    $PY tools/ref/ref_deepseek.py --gguf $M/moonlight-16b/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf --config $M/moonlight-16b/config.json \
        --tokenizer $M/moonlight-16b --out $M/moonlight-16b/moonlight_golden.tfref --generate 32 --threads 10 > $O/ref_moonlight.log 2>&1
    echo "exit $?" > $O/REF_moonlight.done
) &
fi
wait
