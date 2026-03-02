import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from spectra.engine.trainer import SPECTRAModule
from spectra.data.clinical.dataset import ICUTrajectoryDataset

def main():
    print("--- Testing SPECTRAModule with PCGrad on REAL Clinical Sepsis Data ---")
    
    # 1. Create Config matching clinical.yaml
    cfg_dict = {
        "dataset_dir": "datasets/sepsis_clinical_28",
        "model": {
            "backbone": "shared_trunk",
            "input_dim": 28,
            "d_model": 512,
            "hidden_layers": 4,
            "dropout": 0.2
        },
        "method": {
            "name": "pcgrad",
            "use_alb": False,
        },
        "tasks": [
            {"name": "outcome", "type": "classification", "loss": "bce", "num_classes": 1, "manifold": "expert", "weight": 1.0, "pos_weight": 3.0},
            {"name": "phase", "type": "classification", "loss": "cross_entropy", "num_classes": 3, "manifold": "planner", "weight": 1.0}
        ],
        "train": {
            "lr": 5e-5,
            "min_lr": 1e-6,
            "warmup_steps": 10,  # low for testing
            "weight_decay": 1e-4,
            "grad_clip": 1.0,  # <--- using our tightened fix!
            "epochs": 5
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    # 2. Setup Dataset & Dataloader
    # subset_pct=0.05 reads only 5% of the data to make the minimal test run fast
    try:
        train_ds = ICUTrajectoryDataset(dataset_dir="datasets/sepsis_clinical_28", split="train", subset_pct=0.05)
        val_ds = ICUTrajectoryDataset(dataset_dir="datasets/sepsis_clinical_28", split="val", subset_pct=0.05)
    except Exception as e:
        print(f"Skipping test, data not available locally: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # 3. Instantiate Module
    model = SPECTRAModule(cfg)
    
    # 4. Custom Callback for Terminal Logging
    class PrintMetricsCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train/total_loss")
            gn = trainer.callback_metrics.get("health/backbone_grad_norm")
            conflicts = trainer.callback_metrics.get("pcgrad/total_conflicts")
            if train_loss is not None:
                print(f"[Epoch {trainer.current_epoch} Train] Loss: {train_loss:.4f} | GN: {gn:.4f} | Conflicts: {conflicts:.1f}")
            
        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val/total_loss")
            out_loss = trainer.callback_metrics.get("val/outcome_loss")
            auc = trainer.callback_metrics.get("val/outcome_AUC")
            if val_loss is not None:
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                out_loss_str = f"{out_loss:.4f}" if out_loss is not None else "N/A"
                print(f"[Epoch {trainer.current_epoch} Val] Total Loss: {val_loss:.4f} | Outcome Val Loss: {out_loss_str} | Outcome AUC: {auc_str}")

    # 5. Train
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=5,
        callbacks=[PrintMetricsCallback()]
    )
    
    print(f"Starting training on {len(train_ds)} samples...")
    trainer.fit(model, train_loader, val_loader)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
