<#
.SYNOPSIS
SPECTRA Synthetic Benchmark Runner (PowerShell)

.DESCRIPTION
This script executes the full 80-epoch convergence gauntlet on the Synthetic dataset
across all six core multi-task learning baselines. 

.NOTES
- Ensures sequential execution.
- Automatically creates descriptive CSV logs in the outputs/ directory.
- Aborts on error to prevent cascading failures.
#>

$ErrorActionPreference = "Stop"

$METHODS = @("static", "kendall", "uwso", "ntkmtl", "pcgrad", "bpgs", "bpgs_alb")
$EPOCHS = 100
$DATASET = "synthetic"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SPECTRA Synthetic Dataset Gauntlet ($EPOCHS Epochs)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($method in $METHODS) {
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Launching: $method" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    
    # Run the training script, overriding the epochs to 80 for the gauntlet
    # We leave train.save_ckpt=true (default) to ensure we save the best models
    python scripts/train.py dataset=$DATASET method=$method train.epochs=$EPOCHS

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] Gauntlet halted! Baseline '$method' failed." -ForegroundColor Red
        Write-Host "Please check the terminal output for the stack trace." -ForegroundColor Red
        exit 1
    }

    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Success: $method completed." -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " GAUNTLET COMPLETE! All baselines executed successfully." -ForegroundColor Cyan
Write-Host " Logs are available in: ./outputs/" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
