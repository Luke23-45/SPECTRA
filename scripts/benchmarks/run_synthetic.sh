#!/usr/bin/env bash
# ==============================================================
# SYNOPSIS
#   SPECTRA Synthetic Benchmark Runner (Bash)
#
# DESCRIPTION
#   This script executes the synthetic convergence gauntlet
#   on the Synthetic dataset across the active SPECTRA method set.
#
# NOTES
#   - Ensures sequential execution.
#   - Automatically creates descriptive CSV logs in outputs/.
#   - Aborts on error to prevent cascading failures.
# ==============================================================

# Abort immediately if any command exits with a non-zero status,
# if an unset variable is referenced, or if a pipe fails.
set -euo pipefail

# --------------------------------------------------------------
# ANSI colour helpers
# --------------------------------------------------------------
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'   # No Colour / reset

# --------------------------------------------------------------
# Configuration  (mirror the PowerShell variables exactly)
# --------------------------------------------------------------
METHODS=("static" "kendall" "uwso" "gradnorm_proxy" "pcgrad" "bpgs")
EPOCHS=80
DATASET="synthetic"

# --------------------------------------------------------------
# Helper: current timestamp in the same format as PowerShell
# --------------------------------------------------------------
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# --------------------------------------------------------------
# Banner
# --------------------------------------------------------------
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} SPECTRA Synthetic Dataset Gauntlet (${EPOCHS} Epochs)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# --------------------------------------------------------------
# Main loop
# --------------------------------------------------------------
for method in "${METHODS[@]}"; do
    echo -e "${YELLOW}------------------------------------------------------------${NC}"
    echo -e "${YELLOW}[$(timestamp)] Launching: ${method}${NC}"
    echo -e "${YELLOW}------------------------------------------------------------${NC}"

    # Run the training script, overriding epochs for the gauntlet.
    # train.save_ckpt=true (default) ensures the best models are saved.
    python scripts/train.py \
        dataset="$DATASET" \
        method="$method" \
        train.epochs="$EPOCHS"

    # If we reach here, the previous command succeeded.

    echo -e "${GREEN}[$(timestamp)] Success: ${method} completed.${NC}"
    echo ""
done

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} GAUNTLET COMPLETE! All baselines executed successfully.${NC}"
echo -e "${CYAN} Logs are available in: ./outputs/${NC}"
echo -e "${CYAN}============================================================${NC}"
