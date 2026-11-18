# TASK-32: layer 3's attention and MoE inputs and outputs on the short prompts, to locate the engine's difference.
$env:PYTHONUNBUFFERED = '1'
$m = 'E:\models\glm53-flash'
& "$m\ref-venv\Scripts\python.exe" D:\CLionProjects\DistributedMoEEngine\tools\ref\ref_glm53.py `
    --gguf "$m\GLM-5.3-Flash-Q4_K-00001-of-00002.gguf" "$m\GLM-5.3-Flash-Q4_K-00002-of-00002.gguf" `
    --config "$m\config.json" --tokenizer "$m\tokenizer.json" --out "$m\glm53_layer3.tfref" `
    --lazy --dump-layer 3 2>&1 |
    ForEach-Object { "$(Get-Date -Format HH:mm:ss) $_" } | Out-File -Encoding utf8 "$m\reference_l3.log"
"REF_EXIT $LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 "$m\reference_l3.log"
