"""
scripts/data/qm9_generate.py
----------------------------
Thin wrapper around the QM9 ingestion pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectra.data.qm9.ingest import main

if __name__ == "__main__":
    main()
