"""
scripts/data/nyuv2_generate.py
-------------------------------
Dedicated data generation script for the NYUv2 pipeline.

This is a thin wrapper around the production pipeline defined in
spectra.data.nyuv2.nyuv2_lmdb_sota.run_pipeline(). All logic lives
in that module — this script simply delegates to it.

Usage:
    python scripts/execution/experiment_runner.py nyuv2_generate
    python scripts/data/nyuv2_generate.py --limit 10
    python scripts/data/nyuv2_generate.py --force
    python scripts/data/nyuv2_generate.py --keep-staging
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so spectra is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectra.data.nyuv2.nyuv2_lmdb_sota import main

if __name__ == "__main__":
    main()
