"""
spectra/utils/progress.py
-------------------------
Research-Grade Telemetry and Progress Interfaces.
"""

from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

class SOTAProgressBar(TQDMProgressBar):
    """
    Research-Grade Progress Bar (SOTA Style).
    Maps long metric keys to concise research shorthand (GN, L, AUC, etc.).
    """
    def init_train_tqdm(self) -> tqdm:
        bar = super().init_train_tqdm()
        # "Gold Standard" Format: Dense, no bars, high-fidelity telemetry
        bar.bar_format = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = super().init_validation_tqdm()
        bar.bar_format = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None) # Remove v_num to save space
        
        # SOTA Shorthand Mapping
        mapping = {
            "train/total_loss": "L",
            "health/backbone_grad_norm": "GN",
            "health/update_weight_ratio": "Ratio",
            "pcgrad/total_conflicts": "C",
            "spectral/hf_divorce_index": "D",
            "health/backbone_weight_norm": "WN",
            "val/total_loss": "vL",
            "train/AUC": "AUC",
            "train/PRC": "PRC",
            "train/R": "R",
        }
        
        # Add task-specific adaptive mapping
        for task in getattr(pl_module, "task_names", []):
            mapping[f"train/{task}_loss"] = f"L_{task[:1]}"
            mapping[f"val/{task}_miou"] = "mIoU"
            mapping[f"val/{task}_abs_rel"] = "abs_rel"
            mapping[f"val/{task}_mean_angle"] = "angle"
            
        new_items = {}
        for k, v in items.items():
            base_k = k.replace("_step", "").replace("_epoch", "")
            # Priority 1: Explicit mapping (with prefix)
            if base_k in mapping:
                new_items[mapping[base_k]] = v
            # Priority 2: Explicit mapping (without prefix)
            elif base_k.split("/")[-1] in [m.split("/")[-1] for m in mapping if "/" in m]:
                # Find the shorthand for the base key
                for mk, mv in mapping.items():
                    if mk.endswith(f"/{base_k.split('/')[-1]}"):
                        new_items[mv] = v
                        break
            else:
                clean_k = k.replace("train/", "").replace("val/", "").replace("health/", "")
                new_items[clean_k] = v
        return new_items
