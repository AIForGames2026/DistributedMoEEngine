# TASK-32 sensitivity: the reference with its embeddings nudged by 1e-6 relative noise, compared with the plain run.
$env:PYTHONUNBUFFERED = '1'
$m = 'E:\models\glm53-flash'
& "$m\ref-venv\Scripts\python.exe" D:\CLionProjects\DistributedMoEEngine\tools\ref\ref_glm53.py `
    --gguf "$m\GLM-5.3-Flash-Q4_K-00001-of-00002.gguf" "$m\GLM-5.3-Flash-Q4_K-00002-of-00002.gguf" `
    --config "$m\config.json" --tokenizer "$m\tokenizer.json" --out "$m\glm53_golden_p6.tfref" `
    --lazy --long "$m\long_prompt.txt" --long-tokens 2600 --tail 32 --perturb 1e-6 2>&1 |
    ForEach-Object { "$(Get-Date -Format HH:mm:ss) $_" } | Out-File -Encoding utf8 "$m\reference_p6.log"
"REF_EXIT $LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 "$m\reference_p6.log"
