import sys
import os

# Set up project path
project_path = r"C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\SPECTRA"
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from tests.verify_bpgs_pure import test_bpgs_pure_gradients

if __name__ == "__main__":
    try:
        test_bpgs_pure_gradients()
        print("\nUnit Test Summary: ALL PASS")
    except Exception as e:
        print(f"\nUnit Test Summary: FAILED\nError: {e}")
        sys.exit(1)
