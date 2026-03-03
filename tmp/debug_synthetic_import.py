import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))

try:
    from spectra.modules.synthetic import SyntheticSPECTRAModule
    print("SUCCESS: SyntheticSPECTRAModule imported successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
