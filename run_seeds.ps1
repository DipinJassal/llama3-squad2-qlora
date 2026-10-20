# Seed replicates for 3 grid configs, to measure run-to-run variance. Same settings as the
# original grid runs (qv used batch 16x1, qkvo/all used 8x2). Skips finished runs.
$env:HF_TOKEN = (Get-Content "$HOME\.cache\huggingface\token" -Raw).Trim()
New-Item -ItemType Directory -Force logs | Out-Null

$configs = @(
    @{ modules = 'qv';  rank = 8;  batch = @('--batch-size', '16', '--grad-accum', '1') },
    @{ modules = 'all'; rank = 16; batch = @() },
    @{ modules = 'all'; rank = 64; batch = @() }
)
foreach ($seed in 1, 2) {
    foreach ($c in $configs) {
        $name = "$($c.modules)-r$($c.rank)-s$seed"
        if (Test-Path "adapters\$name\train_info.json") { Write-Output "$name already trained, skipping"; continue }
        Write-Output "$(Get-Date -Format 'HH:mm') training $name"
        docker run --rm --gpus all --ipc=host -e HF_TOKEN `
            -v hf-cache:/root/.cache/huggingface -v "${PWD}:/work" squad-train `
            python3 train.py --modules $c.modules --rank $c.rank --n-train 10000 --seed $seed @($c.batch) *> "logs\$name.log"
        Write-Output "$(Get-Date -Format 'HH:mm') finished $name (exit $LASTEXITCODE)"
    }
}
