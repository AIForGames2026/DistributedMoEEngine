# TASK-32 reference: transformers' GLM-5.3-Flash on the Q4_K GGUF, large matrices kept quantized (--lazy).
$env:PYTHONUNBUFFERED = '1'
$m = 'E:\models\glm53-flash'
& "$m\ref-venv\Scripts\python.exe" D:\CLionProjects\DistributedMoEEngine\tools\ref\ref_glm53.py `
    --gguf "$m\GLM-5.3-Flash-Q4_K-00001-of-00002.gguf" "$m\GLM-5.3-Flash-Q4_K-00002-of-00002.gguf" `
    --config "$m\config.json" --tokenizer "$m\tokenizer.json" --out "$m\glm53_golden.tfref" `
    --generate 32 --lazy --long "$m\long_prompt.txt" --long-tokens 2600 --tail 32 2>&1 |
    ForEach-Object { "$_" } | Out-File -Encoding utf8 "$m\reference.log"
"REF_EXIT $LASTEXITCODE" | Out-File -Append -Encoding utf8 "$m\reference.log"
