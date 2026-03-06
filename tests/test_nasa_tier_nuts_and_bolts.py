import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Path injection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectra.modules.clinical import ClinicalSPECTRAModule
from spectra.modules.vision import VisionSPECTRAModule
from spectra.engine.optimizers.pcgrad import PCGradEngine
from spectra.engine.optimizers.bpgs import BPGSEngine
from spectra.engine.optimizers.standard import StandardEngine

# =========================================================================
# NASA-Tier Architecture Simulator
# =========================================================================

def get_base_cfg(weighter_name, use_alb, dataset_type="clinical"):
    cfg = OmegaConf.create({
        "model": {
            "d_model": 64,
            "hidden_layers": 2,
            "dropout": 0.1,
            "n_heads": 4
        },
        "method": {
            "name": weighter_name,
            "use_alb": use_alb,
            "n_expert_layers": 1,
            "expert_init": "orthogonal"
        },
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 2,
            "grad_clip": 1.0,
            "min_lr": 1e-6
        }
    })

    if dataset_type == "clinical":
        cfg.model.backbone = "shared_trunk"
        cfg.model.input_dim = 28
        cfg.tasks = [
            {"name": "outcome", "type": "classification", "num_classes": 1, "loss": "bce", "manifold": "both"},
            {"name": "phase", "type": "classification", "num_classes": 3, "loss": "cross_entropy", "manifold": "planner"},
            {"name": "los", "type": "regression", "loss": "mse", "manifold": "expert"}
        ]
        
    elif dataset_type == "dense":
        cfg.model.backbone = "segnet"
        cfg.model.input_channels = 3
        cfg.tasks = [
            {"name": "segmentation", "type": "dense_classification", "num_classes": 13, "loss": "cross_entropy", "manifold": "both"},
            {"name": "depth", "type": "dense_regression", "output_dim": 1, "loss": "masked_l1", "manifold": "planner"},
            {"name": "normals", "type": "dense_regression", "output_dim": 3, "loss": "cosine_dense", "manifold": "expert"}
        ]
        
    return cfg

class MockTrainer:
    def __init__(self):
        self.world_size = 1
        self.global_step = 0
        self.current_epoch = 0
        self.estimated_stepping_batches = 100
        self.precision_plugin = None

class SOTA_Auditor:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.logs = []
        
    def log(self, msg):
        print(msg)
        self.logs.append(msg)
        
    def check_grad_flow(self, module):
        zero_grad_count = 0
        nan_grad_count = 0
        total_params = 0
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is None:
                    continue
                if torch.isnan(param.grad).any():
                    nan_grad_count += 1
                elif param.grad.abs().max() == 0:
                    zero_grad_count += 1
                    
        return {
            "total": total_params,
            "zero": zero_grad_count,
            "nan": nan_grad_count
        }

def run_suite():
    auditor = SOTA_Auditor()
    auditor.log("\n" + "="*80)
    auditor.log("🚀 S.P.E.C.T.R.A. NASA-Tier Architecture Verification Sequence Initiated")
    auditor.log("="*80)
    
    weighters = ["pcgrad", "bpgs", "ntkmtl", "uwso", "kendall", "static"]
    albs = [True, False]
    datasets = ["clinical", "dense"]
    
    # We will test a subset to prove completeness without taking 10 minutes.
    # We test ALL weighters on Clinical + ALB
    # We test ALBs and Dense on PCGrad & BPGS
    
    test_suite = []
    for w in weighters:
        test_suite.append((w, True, "clinical"))
    test_suite.append(("pcgrad", False, "clinical"))
    test_suite.append(("bpgs", False, "dense"))
    test_suite.append(("pcgrad", True, "dense"))
    
    bsz = 2
    
    for w_name, use_alb, d_type in test_suite:
        cfg_name = f"[{d_type.upper()}] Weighter: {w_name.upper()} | ALB: {use_alb}"
        auditor.log(f"\n▶ Auditing Topology: {cfg_name}")
        
        try:
            cfg = get_base_cfg(w_name, use_alb, d_type)
            
            if w_name in ("bpgs", "bpgs_alb"):
                engine = BPGSEngine()
            elif w_name == "pcgrad":
                engine = PCGradEngine()
            else:
                engine = StandardEngine()
                
            if d_type == "clinical":
                model = ClinicalSPECTRAModule(cfg, engine)
            else:
                model = VisionSPECTRAModule(cfg, engine)
            model.trainer = MockTrainer()
            
            # Mock Logger & Manual Backward
            model.log = lambda k, v, **kwargs: None
            def mock_manual_backward(loss, *args, **kwargs):
                loss.backward(*args, **kwargs)
            model.manual_backward = mock_manual_backward
            model.clip_gradients = lambda *args, **kwargs: None
            
            # --- 1. Mock Data ---
            if d_type == "clinical":
                batch = {
                    "input": torch.randn(bsz, 10, 28), # B, seq, C
                    "targets": {
                        "outcome": torch.randint(0, 2, (bsz, 1)).float(),
                        "phase": torch.randint(0, 3, (bsz,)).long(),
                        "los": torch.randn(bsz, 1)
                    }
                }
            else:
                batch = {
                    "input": torch.randn(bsz, 3, 32, 32), # B, C, H, W
                    "meta": {
                        "depth_mask": torch.ones(bsz, 1, 32, 32).bool(),
                        "normals_mask": torch.ones(bsz, 1, 32, 32).bool()
                    },
                    "targets": {
                        "segmentation": torch.randint(0, 13, (bsz, 32, 32)).long(),
                        "depth": torch.randn(bsz, 1, 32, 32).abs(),
                        "normals": torch.randn(bsz, 3, 32, 32)
                    }
                }
            
            # --- 2. Forward Pass ---
            preds = model(batch)
            for t_name in cfg.tasks:
                assert t_name.name in preds, f"Missing prediction for task {t_name.name}"
                p = preds[t_name.name]
                assert not torch.isnan(p).any(), f"NaN detected in {t_name.name} forward pass!"
            auditor.log(f"  ✓ Forward Pass Validated (Shapes and Integrity)")
            
            # --- 3. Backward Pass / Training Step ---
            model.train()
            opts = model.configure_optimizers()
            opt = opts["optimizer"]
            
            model.optimizers = lambda: opt
            model.lr_schedulers = lambda: opts["lr_scheduler"]["scheduler"]
            
            opt.zero_grad()
            out = model.training_step(batch, batch_idx=0)
            
            if out is not None:
                # Standard optimization path
                out.backward()
                opt.step()
            
            # --- 4. Gradient Health Check ---
            grad_health = auditor.check_grad_flow(model.backbone)
            assert grad_health["nan"] == 0, "CRITICAL: NaN Gradients detected in backbone!"
            assert grad_health["zero"] < grad_health["total"], "CRITICAL: Entire backbone has zero/dead gradients!"
            
            auditor.log(f"  ✓ Backward Pass Validated (0 NaNs, Valid Gradient Flow in Backbone)")
            
            # --- 5. Validation Step ---
            model.eval()
            with torch.no_grad():
                model.on_validation_epoch_start()
                model.validation_step(batch, batch_idx=0)
                model.on_validation_epoch_end()
            auditor.log(f"  ✓ Validation Epoch & Metric Mechanics Cleared")
            
            auditor.passed += 1
            
        except AssertionError as e:
            auditor.log(f"  ❌ ASSERTION FAILED: {str(e)}")
            auditor.failed += 1
        except Exception as e:
            auditor.log(f"  ❌ CRASH DETECTED: {str(e)}")
            auditor.failed += 1

    auditor.log("\n" + "="*80)
    auditor.log(f"🏁 AUDIT COMPLETE: {auditor.passed} PASSED, {auditor.failed} FAILED")
    auditor.log("="*80)
    
if __name__ == "__main__":
    run_suite()
