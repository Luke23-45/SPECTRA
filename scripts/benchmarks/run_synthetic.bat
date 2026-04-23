@echo off
setlocal enabledelayedexpansion

:: ========================================================================
:: SPECTRA Synthetic Benchmark Runner (Batch)
::
:: This script executes the synthetic convergence gauntlet on the
:: Synthetic dataset across the active SPECTRA method set.
:: ========================================================================

set METHODS=static kendall uwso gradnorm_proxy pcgrad bpgs
set EPOCHS=100
set DATASET=synthetic

echo ============================================================
echo  SPECTRA Synthetic Dataset Gauntlet (%EPOCHS% Epochs)
echo ============================================================
echo.

for %%M in (%METHODS%) do (
    echo ------------------------------------------------------------
    echo Launching: %%M
    echo ------------------------------------------------------------
    
    :: Run the training script, overriding the epochs to 80 for the gauntlet
    python scripts/train.py dataset=%DATASET% method=%%M train.epochs=%EPOCHS%
    
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [ERROR] Gauntlet halted! Baseline '%%M' failed.
        echo Please check the terminal output for the stack trace.
        exit /b 1
    )
    
    echo Success: %%M completed.
    echo.
)

echo ============================================================
echo  GAUNTLET COMPLETE! All baselines executed successfully.
echo  Logs are available in: ./outputs/
echo ============================================================
exit /b 0
