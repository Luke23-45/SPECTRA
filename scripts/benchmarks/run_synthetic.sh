#!/bin/bash

# ========================================================================
# SPECTRA Synthetic Benchmark Runner (Linux/Colab)
#
# This script executes the full 100-epoch convergence gauntlet on the 
# Synthetic dataset across all six core multi-task learning baselines. 
# ========================================================================

METHODS=("static" "kendall" "uwso" "ntkmtl" "pcgrad" "bpgs")
EPOCHS=100
DATASET="synthetic"

echo "============================================================"
echo " SPECTRA Synthetic Dataset Gauntlet ($EPOCHS Epochs)"
echo " Environment: Google Colab / Linux"
echo "============================================================"
echo

for METHOD in "${METHODS[@]}"
do
    echo "------------------------------------------------------------"
    echo " Launching: $METHOD"
    echo "------------------------------------------------------------"
    
    # Run the training script, overriding the epochs to 100 for the gauntlet
    # Adding --multirun or hydra syntax if needed, but simple overrides work.
    python scripts/train.py dataset=$DATASET method=$METHOD train.epochs=$EPOCHS run_name="colab_${METHOD}_100e"
    
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Gauntlet halted! Baseline '$METHOD' failed."
        echo "Please check the terminal output for the stack trace."
        exit 1
    fi
    
    echo "Success: $METHOD completed."
    echo
done

echo "============================================================"
echo " GAUNTLET COMPLETE! All baselines executed successfully."
echo " Logs are available in: ./outputs/"
echo "============================================================"
exit 0
