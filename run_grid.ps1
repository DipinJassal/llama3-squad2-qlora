# Train the full ablation grid (target modules x rank), one container per config.
# Skips configs that already finished a full run, so it can be re-launched after an interruption.
param([int]$NTrain = 10000)

$env:HF_TOKEN = (Get-Content "$HOME\.cache\huggingface\token" -Raw).Trim()
New-Item -ItemType Directory -Force logs | Out-Null

foreach ($modules in 'qv', 'qkvo', 'all') {
    foreach ($rank in 8, 16, 32, 64) {
        $name = "$modules-r$rank"
        $info = "adapters\$name\train_info.json"
        if ((Test-Path $info) -and ((Get-Content $info -Raw | ConvertFrom-Json).max_steps -eq -1)) {
            Write-Output "$name already trained, skipping"; continue
        }
        Write-Output "$(Get-Date -Format 'HH:mm') training $name"
        docker run --rm --gpus all --ipc=host -e HF_TOKEN `
            -v hf-cache:/root/.cache/huggingface -v "${PWD}:/work" squad-train `
            python3 train.py --modules $modules --rank $rank --n-train $NTrain *> "logs\$name.log"
        Write-Output "$(Get-Date -Format 'HH:mm') finished $name (exit $LASTEXITCODE)"
    }
}
