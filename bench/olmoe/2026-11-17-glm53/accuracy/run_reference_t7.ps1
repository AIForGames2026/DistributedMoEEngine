# TASK-32 noise floor: the same reference with 7 threads instead of 20, prompts only, to compare against itself.
$env:PYTHONUNBUFFERED = '1'
$m = 'E:\models\glm53-flash'
& "$m\ref-venv\Scripts\python.exe" D:\CLionProjects\DistributedMoEEngine\tools\ref\ref_glm53.py `
    --gguf "$m\GLM-5.3-Flash-Q4_K-00001-of-00002.gguf" "$m\GLM-5.3-Flash-Q4_K-00002-of-00002.gguf" `
    --config "$m\config.json" --tokenizer "$m\tokenizer.json" --out "$m\glm53_golden_t7.tfref" `
    --lazy --long "$m\long_prompt.txt" --long-tokens 2600 --tail 32 --threads 7 2>&1 |
    ForEach-Object { "$(Get-Date -Format HH:mm:ss) $_" } | Out-File -Encoding utf8 "$m\reference_t7.log"
"REF_EXIT $LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 "$m\reference_t7.log"
