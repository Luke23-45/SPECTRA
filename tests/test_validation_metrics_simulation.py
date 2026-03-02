import os
import sys
import torch
from omegaconf import OmegaConf

# Ensure path is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectra.engine.trainer import SPECTRAModule

def mock_clinical_config():
    return OmegaConf.create({
        "model": {
            "backbone": "shared_trunk",
            "input_dim": 28,
            "d_model": 64,
            "hidden_layers": 2,
            "dropout": 0.1,
            "n_heads": 4
        },
        "method": {
            "name": "pcgrad",
            "use_alb": True,
            "n_expert_layers": 1,
            "expert_init": "orthogonal"
        },
        "tasks": [
            {
                "name": "outcome",
                "type": "classification",
                "num_classes": 1,
                "loss": "bce",
                "manifold": "both",
                "weight": 2.0
            },
            {
                "name": "phase",
                "type": "classification",
                "num_classes": 3,
                "loss": "cross_entropy",
                "manifold": "planner",
                "weight": 0.5
            }
        ],
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 10
        }
    })

class MockTrainer:
    def __init__(self):
        self.world_size = 1
        self.logged_metrics = {}
        self.precision_plugin = None

def run_simulation():
    print("=== Starting Recursive Validation Metrics Simulation ===")
    
    # Subproblem 1: Initialization
    print("\n[1] Testing Initialization & Metric Binding...")
    cfg = mock_clinical_config()
    model = SPECTRAModule(cfg)
    model.trainer = MockTrainer()
    
    # Mock the log function
    def mock_log(name, value, **kwargs):
        model.trainer.logged_metrics[name] = value.item() if isinstance(value, torch.Tensor) else value
    
    model.log = mock_log
    
    assert "outcome" in model._val_metrics, "Outcome metric missing!"
    assert "phase" in model._val_metrics, "Phase metric missing!"
    print(" ✓ Metrics Initialized Correctly")

    # Subproblem 2: Shapes & Accumulation (validation_step)
    print("\n[2] Testing Shape Alignment in validation_step...")
    batch_size = 4
    seq_len = 10
    
    # Mock Batch
    batch = {
        "input": torch.randn(batch_size, seq_len, 28),
        "targets": {
            "outcome": torch.randint(0, 2, (batch_size, 1)).float(), # [B, 1] label
            "phase": torch.randint(0, 3, (batch_size,)).long()      # [B] label
        }
    }
    
    # Model forward pass
    preds = model(batch)
    assert preds["outcome"].shape == (batch_size, 1), f"Outcome pred shape mismatch: {preds['outcome'].shape}"
    assert preds["phase"].shape == (batch_size, 3), f"Phase pred shape mismatch: {preds['phase'].shape}"
    
    # We must explicitly set model to eval mode!
    model.eval()
    with torch.no_grad():
        model.on_validation_epoch_start()
        model.validation_step(batch, batch_idx=0)
    
    print(" ✓ validation_step executed without shape crashes.")

    # Verify weighted total validation loss is respected
    out_l = model.trainer.logged_metrics["val/outcome_loss"]
    phase_l = model.trainer.logged_metrics["val/phase_loss"]
    total_l = model.trainer.logged_metrics["val/total_loss"]
    expected_total = 2.0 * out_l + 0.5 * phase_l
    assert abs(total_l - expected_total) < 1e-5, (
        f"Weighted val total loss mismatch: got={total_l}, expected={expected_total}"
    )
    print(" ✓ Weighted validation total loss verified.")
    
    # Check internal metric states
    out_state = model._val_metrics["outcome"]
    # Internal variables in torchmetrics vary, but we can check if it has been updated
    assert len(out_state["AUC"].target) > 0, "Outcome AUC state list is empty, accumulation failed."
    print(" ✓ Metric accumulation verified.")

    # Subproblem 3: Epoch End Computations (on_validation_epoch_end)
    print("\n[3] Testing Epoch Aggregation & Logging...")
    with torch.no_grad():
        model.on_validation_epoch_end()
    
    logs = model.trainer.logged_metrics
    required_keys = [
        "val/outcome_AUC", 
        "val/outcome_PRC", 
        "val/outcome_R",
        "val/AUC", 
        "val/phase_ACC", 
        "val/phase_AUC"
    ]
    
    for k in required_keys:
        assert k in logs, f"CRITICAL FAILURE: {k} missing from validation logs!"
        print(f" ✓ Logged {k}: {logs[k]:.4f}")
        
    print("\n=== All Recursive Simulation Checks PASSED ===")

if __name__ == "__main__":
    run_simulation()
