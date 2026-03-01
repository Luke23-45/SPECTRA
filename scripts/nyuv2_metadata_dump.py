"""
scripts/nyuv2_metadata_dump.py
-------------------------------
Forensic Metadata Auditor for NYUv2 (tanganke/nyuv2).

Extracts all non-spatial metadata fields and dumps them to a CSV 
for absolute architectural transparency.
"""

import os
import csv
import logging
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Metadata-Auditor")

def dump_metadata(repo="tanganke/nyuv2", output_file="datasets/nyuv2_metadata_audit.csv"):
    logger.info(f"Connecting to {repo} (Streaming Mode)...")
    ds = load_dataset(repo, streaming=True)
    
    # 1. Identify all keys in the first sample
    first_sample = next(iter(ds['train']))
    all_keys = list(first_sample.keys())
    
    # We exclude the known large spatial arrays to keep the CSV readable
    spatial_keys = {'image', 'segmentation', 'depth', 'normal', 'noise'}
    metadata_keys = [k for k in all_keys if k not in spatial_keys]
    
    # If there's a 'noise' field, we'll inspect its type. If it's a scalar or string, we keep it.
    # From previous check, 'noise' was a list (likely a spatial map), but we'll be thorough.
    
    logger.info(f"Available Keys: {all_keys}")
    logger.info(f"Auditing Metadata Keys: {metadata_keys}")
    
    if not metadata_keys:
        logger.warning("No explicit text/scalar metadata keys found in the top-level schema.")
        # We will still create a CSV with the spatial integrity check (shapes)
        metadata_keys = ["sample_idx", "image_shape", "noise_shape"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata_keys)
        writer.writeheader()
        
        # We audit the first 100 samples of available splits for consistency
        for split in ds.keys():
            logger.info(f"Auditing split: {split}")
            count = 0
            for i, sample in enumerate(tqdm(ds[split], desc=f"Auditing {split}")):
                if i >= 100: break # Focused audit
                
                row = {}
                for k in metadata_keys:
                    if k == "sample_idx":
                        row[k] = i
                    elif k == "image_shape":
                        row[k] = len(sample['image'])
                    elif "shape" in k:
                        row[k] = len(sample.get(k.replace("_shape", ""), []))
                    else:
                        row[k] = sample.get(k, "N/A")
                
                writer.writerow(row)
                count += 1
            logger.info(f"Audit of {split} complete ({count} samples).")

    logger.info(f"Forensic metadata report saved to: {output_file}")

if __name__ == "__main__":
    dump_metadata()
