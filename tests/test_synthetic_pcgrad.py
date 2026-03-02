import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from spectra.engine.trainer import SPECTRAModule
from spectra.data.synthetic import SyntheticMTLDataset

def main():
    print("--- Testing SPECTRAModule with PCGrad on SyntheticMTLDataset ---")
    
    # 1. Create a minimal Hydra-like Config
    cfg_dict = {
        "model": {
            "backbone": "shared_trunk",
            "input_dim": 20,
            "d_model": 64,
            "hidden_layers": 2,
            "dropout": 0.1
        },
        "method": {
            "name": "pcgrad",
            "use_alb": False,
        },
        "tasks": [
            {"name": "mse_high", "type": "regression", "loss": "mse", "manifold": "expert", "weight": 1.0},
            {"name": "bce_0", "type": "classification", "loss": "bce", "num_classes": 1, "manifold": "planner", "weight": 1.0},
            {"name": "outcome", "type": "classification", "loss": "bce", "num_classes": 1, "manifold": "planner", "weight": 1.0}
        ],
        "train": {
            "lr": 1e-3,
            "min_lr": 1e-5,
            "warmup_steps": 10,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "epochs": 50
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    # 2. Setup Dataset & Dataloader
    # We will just use 3 tasks matching the config above
    custom_tasks = [
        {"name": "mse_high", "type": "mse", "scale": 10.0, "offset": 0.0},
        {"name": "bce_0", "type": "bce", "scale": 1.0, "offset": 0.0},
        {"name": "outcome", "type": "bce", "scale": 1.0, "offset": 0.0} # Named outcome to trigger label smoothing
    ]
    
    train_ds = SyntheticMTLDataset(n_samples=500, input_dim=20, hidden_dim=64, task_configs=custom_tasks)
    val_ds = SyntheticMTLDataset(n_samples=100, input_dim=20, hidden_dim=64, task_configs=custom_tasks)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=SyntheticMTLDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=SyntheticMTLDataset.collate_fn)
    
    # 3. Instantiate Module
    model = SPECTRAModule(cfg)
    
    class PrintLossCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train/total_loss")
            print(f"[Epoch {trainer.current_epoch}] Train Loss: {train_loss:.4f}" if train_loss else "")
            
        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val/total_loss")
            out_loss = trainer.callback_metrics.get("val/outcome_loss")
            print(f"[Epoch {trainer.current_epoch} Val] Total Val Loss: {val_loss:.4f} | Outcome Val Loss: {out_loss:.4f}" if val_loss else "")

    # 4. Train
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=5,
        callbacks=[PrintLossCallback()]
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
