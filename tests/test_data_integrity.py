import json
import os
import lmdb
import numpy as np

def check_data_integrity(data_dir="datasets/sepsis_clinical_28"):
    print(f"--- Checking APEX Data Integrity: {data_dir} ---")
    
    index_path = os.path.join(data_dir, "train_index.json")
    lmdb_path = os.path.join(data_dir, "train", "data.lmdb")
    
    with open(index_path, 'r') as f:
        idx_data = json.load(f)
    print(f"Loaded {len(idx_data)} episodes from index.")
    
    env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
    
    nans = 0
    infs = 0
    extremes = 0
    outcome_1 = 0
    outcome_0 = 0
    
    global_min = float('inf')
    global_max = float('-inf')
    
    ep_ids = list(idx_data.keys())[:max_samples]
    
    with env.begin() as txn:
        for ep_id in ep_ids:
            meta = idx_data[ep_id]
            vitals_meta = meta.get("modalities", {}).get("vitals")
            labels_meta = meta.get("modalities", {}).get("labels")
            
            if not vitals_meta or not labels_meta:
                continue
                
            # Read Vitals
            raw_v = txn.get(vitals_meta["key"].encode("ascii"))
            vitals = np.frombuffer(raw_v, dtype=np.dtype(vitals_meta["dtype"])).reshape(vitals_meta["shape"])
            
            # Constraints Check
            if np.isnan(vitals).any(): nans += 1
            if np.isinf(vitals).any(): infs += 1
            
            if vitals.min() < global_min: global_min = vitals.min()
            if vitals.max() > global_max: global_max = vitals.max()
            
            if vitals.max() > 1000 or vitals.min() < -1000:
                extremes += 1
                
            # Read Labels
            raw_l = txn.get(labels_meta["key"].encode("ascii"))
            labels = np.frombuffer(raw_l, dtype=np.dtype(labels_meta["dtype"])).reshape(labels_meta["shape"])
            
            if labels.sum() > 0:
                outcome_1 += 1
            else:
                outcome_0 += 1
                
    print(f"\n--- Integrity Report (n={max_samples}) ---")
    print(f"Episodes with NaNs: {nans}")
    print(f"Episodes with Infs: {infs}")
    print(f"Episodes with Extreme values (>1000 or <-1000): {extremes}")
    print(f"Global Min: {global_min:.4f}")
    print(f"Global Max: {global_max:.4f}")
    
    # Check if the inputs are unnormalized which destroys neural nets
    if global_max > 50 or global_min < -50:
        print("🚨 CRITICAL: The data appears completely UNNORMALIZED! This will explode gradients and destroy model training.")
        
    print(f"\nLabel Distribution (Sepsis Positive vs Negative Episodes): {outcome_1} vs {outcome_0}")
    if outcome_1 == 0 or outcome_0 == 0:
        print("🚨 CRITICAL: Sub-sample lacks any class variance! Dataset generation might be broken or boosted maliciously.")
        
if __name__ == "__main__":
    check_data_integrity()
