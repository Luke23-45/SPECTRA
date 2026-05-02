"""
scripts/data/rf1_generate.py
----------------------------
Thin wrapper around the RF1 ingestion pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectra.data.rf1.ingest import main

if __name__ == "__main__":
    main()
