"""
scripts/data/yeast_generate.py
------------------------------
Thin wrapper around the Yeast ingestion pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectra.data.yeast.ingest import main

if __name__ == "__main__":
    main()
