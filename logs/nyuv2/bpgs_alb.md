==========================================================
 Starting Experiment: bpgs_alb_nyuv2
 Type:                hydra
 Start Time:          2026-04-24 07:09:04+00:00
 Hydra Overrides:     dataset=nyuv2 method=bpgs_alb
==========================================================
Seed set to 42
[2026-04-24 07:09:23,202][spectra.runner][INFO] - [Mission-Control] Workspace: /content/SPECTRA/outputs/bpgs_alb_nyuv2_s42
[2026-04-24 07:09:23,202][spectra.runner][INFO] - [Mission-Control] Stable Artifact Dir: /content/SPECTRA/outputs/bpgs_alb_nyuv2_s42
[2026-04-24 07:09:23,206][spectra.runner][INFO] - [Mission-Control] Config:
dataset_name: nyuv2
benchmark: nyuv2
root: datasets/nyuv2_lmdb/data
augmentation: true
normalize_rgb: false
subset_pct: 1.0
subset_seed: 42
num_classes: 13
ignore_index: 255
image_height: 288
image_width: 384
train_size: 795
val_size: 654
dataset:
  name: nyuv2
  benchmark: nyuv2
  root: ${root}
  augmentation: ${augmentation}
  normalize_rgb: ${normalize_rgb}
  subset_pct: ${subset_pct}
  subset_seed: ${subset_seed}
  num_classes: ${num_classes}
  ignore_index: ${ignore_index}
  image_height: ${image_height}
  image_width: ${image_width}
  train_size: ${train_size}
  val_size: ${val_size}
module:
  _target_: spectra.modules.VisionSPECTRAModule
  engine:
    _target_: spectra.engine.optimizers.BPGSEngine
model:
  backbone: segnet
  input_channels: 3
  d_model: 64
  n_heads: 4
  dropout: 0.1
tasks:
- name: segmentation
  type: dense_classification
  loss: cross_entropy
  num_classes: 13
  ignore_index: 255
  manifold: planner
  weight: 1.0
- name: depth
  type: dense_regression
  loss: masked_l1
  output_dim: 1
  manifold: planner
  weight: 1.0
- name: normals
  type: dense_regression
  loss: cosine_dense
  output_dim: 3
  manifold: expert
  weight: 1.0
train:
  save_ckpt: false
  epochs: 100
  batch_size: 8
  lr: 0.0005
  min_lr: 1.0e-06
  warmup_steps: 100
  weight_decay: 0.0001
  grad_clip: 1.0
  precision: 32
  num_workers: 2
  log_every_n_steps: 10
  deterministic: false
  early_stop: true
  early_stop_patience: 15
  early_stop_min_delta: 0.0001
logging:
  use_wandb: false
  wandb_project: spectra-mtl
  wandb_mode: offline
method_name: ${method.name}
use_alb: true
method:
  name: bpgs_alb
  s_min: -10.0
  s_max: 10.0
  s_init: 0.0
  n_expert_layers: 3
  expert_init: orthogonal
  expert_gain: 1.2
run_name: ${method_name}_${dataset_name}_s${seed}
seed: 42
resume_from: null
output_dir: ./outputs/${run_name}

[2026-04-24 07:09:23,365][spectra.preflight][INFO] - [PreFlight] All systems nominal. GO for training.
[2026-04-24 07:09:23,374][spectra.runner][INFO] - [Mission-Control] Config saved to: /content/SPECTRA/outputs/bpgs_alb_nyuv2_s42/config.yaml
[2026-04-24 07:09:23,390][spectra.runner][INFO] - [Mission-Control] Metadata saved to: /content/SPECTRA/outputs/bpgs_alb_nyuv2_s42/metadata.json
/content/SPECTRA/spectra/engine/scalers.py:20: SyntaxWarning: invalid escape sequence '\m'
  $y_{norm} = (y - \mu) / \sigma$
[2026-04-24 07:09:23,588][spectra.backbones.segnet][INFO] - [SegNet] Initialized: in=3, d_model=64, stages=5, params=18,855,552
[2026-04-24 07:09:25,017][spectra.runner][INFO] - [Logging] CSV Logger initialized in stable artifact dir: /content/SPECTRA/outputs/bpgs_alb_nyuv2_s42/csv_logs/bpgs_alb_nyuv2_s42
[2026-04-24 07:09:25,018][spectra.runner][INFO] - [Mission-Control] Manual optimization active (method=bpgs_alb). Automatic PL gradient clipping disabled — engine manages clipping internally.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
[2026-04-24 07:09:25,073][spectra.runner][INFO] - [Mission-Control] All systems GO. Method=bpgs_alb, Epochs=100, Tasks=['segmentation', 'depth', 'normals']
[2026-04-24 07:09:25,318][spectra.data.nyuv2][INFO] - [NYUv2-TRAIN] Initialized: 795 samples, augmentation=ON, ImageNet_norm=OFF
[2026-04-24 07:09:25,320][spectra.data.nyuv2][INFO] - [NYUv2-VAL] Initialized: 654 samples, augmentation=OFF, ImageNet_norm=OFF
[2026-04-24 07:09:25,320][spectra.datamodule][INFO] - [DataModule] Setup complete: nyuv2 (Train=795, Val=654)
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name         ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model        │ DynamicWrapper │ 18.9 M │ train │     0 │
│ 1 │ backbone     │ SegNet         │ 18.9 M │ train │     0 │
│ 2 │ heads        │ ModuleDict     │ 13.6 K │ train │     0 │
│ 3 │ alb          │ SpatialALB     │ 18.9 M │ train │     0 │
│ 4 │ weighter     │ BPGS           │      3 │ train │     0 │
│ 5 │ task_weights │ ParameterDict  │      3 │ train │     0 │
│ 6 │ task_losses  │ ModuleDict     │      0 │ train │     0 │
│ 7 │ _val_losses  │ ModuleDict     │      0 │ train │     0 │
└───┴──────────────┴────────────────┴────────┴───────┴───────┘
Trainable params: 18.9 M                                                        
Non-trainable params: 3                                                         
Total params: 18.9 M                                                            
Total estimated model params size (MB): 75                                      
Modules in train mode: 108                                                      
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Sanity Checking DataLoader 0:   0% 0/2 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/data.py:79: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 8. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
Training   0% 0/? [00:00<?, ?it/s]         
[GradSpike] Step 150: grad_norm=0.5276 is 8.5x EMA. Watch for instability. (Silencing per-step)
Epoch 0: 100% 99/99 [01:15<00:00,  1.32it/s, L=3.340, GN=0.528]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/data.py:79: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 6. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.

Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.11it/s]
Epoch 0: 100% 99/99 [02:36<00:00,  1.58s/it, L=4.620, GN=0.528, vL=3.520, mIoU=0.0431, miou=0.0431, abs_rel=0.372, angle=48.60]
Epoch 1: 100% 99/99 [01:19<00:00,  1.24it/s, L=4.620, GN=0.132, vL=3.520, mIoU=0.0431, miou=0.0431, abs_rel=0.372, angle=48.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:20,  1.09it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.11it/s]
Epoch 1: 100% 99/99 [02:40<00:00,  1.63s/it, L=3.130, GN=0.132, vL=3.050, mIoU=0.0812, miou=0.0812, abs_rel=0.327, angle=44.20]
Epoch 2: 100% 99/99 [01:19<00:00,  1.24it/s, L=3.130, GN=0.196, vL=3.050, mIoU=0.0812, miou=0.0812, abs_rel=0.327, angle=44.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:37<00:38,  1.08it/s]
Validation DataLoader 0:  73% 60/82 [00:55<00:20,  1.09it/s]
Validation DataLoader 0:  98% 80/82 [01:13<00:01,  1.08it/s]
Validation DataLoader 0: 100% 82/82 [01:15<00:00,  1.09it/s]
Epoch 2: 100% 99/99 [02:41<00:00,  1.63s/it, L=3.020, GN=0.196, vL=3.140, mIoU=0.0859, miou=0.0859, abs_rel=0.332, angle=44.50]
[GradSpike] Step 750: grad_norm=0.4529 is 5.7x EMA. Watch for instability. (Silencing per-step)
Epoch 3: 100% 99/99 [01:19<00:00,  1.24it/s, L=3.020, GN=0.453, vL=3.140, mIoU=0.0859, miou=0.0859, abs_rel=0.332, angle=44.50]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.10it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 3: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.950, GN=0.453, vL=2.940, mIoU=0.090, miou=0.090, abs_rel=0.318, angle=43.50]
[GradSpike] Step 900: grad_norm=0.5380 is 6.3x EMA. Watch for instability. (Silencing per-step)
Epoch 4: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.950, GN=0.146, vL=2.940, mIoU=0.090, miou=0.090, abs_rel=0.318, angle=43.50]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s][2026-04-24 07:22:43,980][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 4: Planner HF=0.5063, Expert HF=0.4697 (Divorce Index: 0.93x)
[2026-04-24 07:22:43,981][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.470) <= Planner HF ratio (0.506). ALB spectral separation not achieved. Check expert branch init.

Epoch 4: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.910, GN=0.146, vL=3.000, mIoU=0.0876, miou=0.0876, abs_rel=0.313, angle=43.20]
Epoch 5: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.910, GN=0.242, vL=3.000, mIoU=0.0876, miou=0.0876, abs_rel=0.313, angle=43.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 5: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.880, GN=0.242, vL=2.910, mIoU=0.0953, miou=0.0953, abs_rel=0.320, angle=43.20]
Epoch 6: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.880, GN=0.238, vL=2.910, mIoU=0.0953, miou=0.0953, abs_rel=0.320, angle=43.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 6: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.880, GN=0.238, vL=2.890, mIoU=0.0932, miou=0.0932, abs_rel=0.317, angle=43.80]
Epoch 7: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.880, GN=0.255, vL=2.890, mIoU=0.0932, miou=0.0932, abs_rel=0.317, angle=43.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 7: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.840, GN=0.255, vL=2.830, mIoU=0.0963, miou=0.0963, abs_rel=0.332, angle=42.70]
Epoch 8: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.840, GN=0.231, vL=2.830, mIoU=0.0963, miou=0.0963, abs_rel=0.332, angle=42.70]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.13it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:36,  1.14it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 8: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.800, GN=0.231, vL=2.900, mIoU=0.0883, miou=0.0883, abs_rel=0.346, angle=42.80]
Epoch 9: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.800, GN=0.277, vL=2.900, mIoU=0.0883, miou=0.0883, abs_rel=0.346, angle=42.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s][2026-04-24 07:36:03,570][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 9: Planner HF=0.4733, Expert HF=0.4309 (Divorce Index: 0.91x)
[2026-04-24 07:36:03,570][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.431) <= Planner HF ratio (0.473). ALB spectral separation not achieved. Check expert branch init.

Epoch 9: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.780, GN=0.277, vL=2.820, mIoU=0.0912, miou=0.0912, abs_rel=0.336, angle=42.30]
Epoch 10: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.780, GN=0.284, vL=2.820, mIoU=0.0912, miou=0.0912, abs_rel=0.336, angle=42.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.14it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 10: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.780, GN=0.284, vL=2.910, mIoU=0.0884, miou=0.0884, abs_rel=0.332, angle=43.20]
Epoch 11: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.780, GN=0.196, vL=2.910, mIoU=0.0884, miou=0.0884, abs_rel=0.332, angle=43.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 11: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.760, GN=0.196, vL=2.810, mIoU=0.103, miou=0.103, abs_rel=0.299, angle=42.10]
Epoch 12: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.760, GN=0.265, vL=2.810, mIoU=0.103, miou=0.103, abs_rel=0.299, angle=42.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 12: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.750, GN=0.265, vL=2.810, mIoU=0.0985, miou=0.0985, abs_rel=0.313, angle=42.30]
Epoch 13: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.750, GN=0.202, vL=2.810, mIoU=0.0985, miou=0.0985, abs_rel=0.313, angle=42.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:20,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 13: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.740, GN=0.202, vL=2.720, mIoU=0.102, miou=0.102, abs_rel=0.305, angle=41.30]
Epoch 14: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.740, GN=0.238, vL=2.720, mIoU=0.102, miou=0.102, abs_rel=0.305, angle=41.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s][2026-04-24 07:49:24,203][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 14: Planner HF=0.4800, Expert HF=0.4180 (Divorce Index: 0.87x)
[2026-04-24 07:49:24,203][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.418) <= Planner HF ratio (0.480). ALB spectral separation not achieved. Check expert branch init.

Epoch 14: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.720, GN=0.238, vL=2.800, mIoU=0.0933, miou=0.0933, abs_rel=0.339, angle=42.10]
Epoch 15: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.720, GN=0.340, vL=2.800, mIoU=0.0933, miou=0.0933, abs_rel=0.339, angle=42.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 15: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.730, GN=0.340, vL=2.690, mIoU=0.102, miou=0.102, abs_rel=0.308, angle=41.30]
Epoch 16: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.730, GN=0.477, vL=2.690, mIoU=0.102, miou=0.102, abs_rel=0.308, angle=41.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 16: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.690, GN=0.477, vL=2.860, mIoU=0.102, miou=0.102, abs_rel=0.393, angle=42.00]
Epoch 17: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.690, GN=0.421, vL=2.860, mIoU=0.102, miou=0.102, abs_rel=0.393, angle=42.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 17: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.670, GN=0.421, vL=2.710, mIoU=0.108, miou=0.108, abs_rel=0.328, angle=40.70]
Epoch 18: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.670, GN=0.266, vL=2.710, mIoU=0.108, miou=0.108, abs_rel=0.328, angle=40.70]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 18: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.680, GN=0.266, vL=2.690, mIoU=0.105, miou=0.105, abs_rel=0.305, angle=41.10]
Epoch 19: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.680, GN=0.118, vL=2.690, mIoU=0.105, miou=0.105, abs_rel=0.305, angle=41.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 08:02:44,175][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 19: Planner HF=0.4485, Expert HF=0.3692 (Divorce Index: 0.82x)
[2026-04-24 08:02:44,175][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.369) <= Planner HF ratio (0.448). ALB spectral separation not achieved. Check expert branch init.

Epoch 19: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.640, GN=0.118, vL=2.770, mIoU=0.117, miou=0.117, abs_rel=0.375, angle=41.00]
Epoch 20: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.640, GN=0.236, vL=2.770, mIoU=0.117, miou=0.117, abs_rel=0.375, angle=41.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:36,  1.14it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 20: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.630, GN=0.236, vL=2.660, mIoU=0.104, miou=0.104, abs_rel=0.282, angle=40.60]
Epoch 21: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.630, GN=0.355, vL=2.660, mIoU=0.104, miou=0.104, abs_rel=0.282, angle=40.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 21: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.630, GN=0.355, vL=2.660, mIoU=0.0994, miou=0.0994, abs_rel=0.283, angle=40.30]
Epoch 22: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.630, GN=0.790, vL=2.660, mIoU=0.0994, miou=0.0994, abs_rel=0.283, angle=40.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.14it/s]
Epoch 22: 100% 99/99 [02:38<00:00,  1.60s/it, L=2.640, GN=0.790, vL=2.650, mIoU=0.111, miou=0.111, abs_rel=0.320, angle=40.80]
Epoch 23: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.640, GN=0.307, vL=2.650, mIoU=0.111, miou=0.111, abs_rel=0.320, angle=40.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 23: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.610, GN=0.307, vL=2.630, mIoU=0.115, miou=0.115, abs_rel=0.313, angle=40.30]
Epoch 24: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.610, GN=0.330, vL=2.630, mIoU=0.115, miou=0.115, abs_rel=0.313, angle=40.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.15it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s][2026-04-24 08:16:00,561][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 24: Planner HF=0.4090, Expert HF=0.3148 (Divorce Index: 0.77x)
[2026-04-24 08:16:00,561][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.315) <= Planner HF ratio (0.409). ALB spectral separation not achieved. Check expert branch init.

Epoch 24: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.610, GN=0.330, vL=2.630, mIoU=0.119, miou=0.119, abs_rel=0.308, angle=40.40]
Epoch 25: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.610, GN=0.264, vL=2.630, mIoU=0.119, miou=0.119, abs_rel=0.308, angle=40.40]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 25: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.590, GN=0.264, vL=2.610, mIoU=0.114, miou=0.114, abs_rel=0.312, angle=40.50]
Epoch 26: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.590, GN=0.398, vL=2.610, mIoU=0.114, miou=0.114, abs_rel=0.312, angle=40.50]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.16it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.14it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 26: 100% 99/99 [02:38<00:00,  1.61s/it, L=2.590, GN=0.398, vL=2.580, mIoU=0.119, miou=0.119, abs_rel=0.286, angle=40.20]
Epoch 27: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.590, GN=0.468, vL=2.580, mIoU=0.119, miou=0.119, abs_rel=0.286, angle=40.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 27: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.570, GN=0.468, vL=2.600, mIoU=0.121, miou=0.121, abs_rel=0.277, angle=40.60]
Epoch 28: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.570, GN=0.380, vL=2.600, mIoU=0.121, miou=0.121, abs_rel=0.277, angle=40.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 28: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.540, GN=0.380, vL=2.570, mIoU=0.127, miou=0.127, abs_rel=0.277, angle=40.10]
Epoch 29: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.540, GN=0.743, vL=2.570, mIoU=0.127, miou=0.127, abs_rel=0.277, angle=40.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 08:29:19,287][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 29: Planner HF=0.3960, Expert HF=0.3022 (Divorce Index: 0.76x)
[2026-04-24 08:29:19,288][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.302) <= Planner HF ratio (0.396). ALB spectral separation not achieved. Check expert branch init.

Epoch 29: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.560, GN=0.743, vL=2.600, mIoU=0.114, miou=0.114, abs_rel=0.293, angle=40.40]
Epoch 30: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.560, GN=0.487, vL=2.600, mIoU=0.114, miou=0.114, abs_rel=0.293, angle=40.40]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 30: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.540, GN=0.487, vL=2.570, mIoU=0.119, miou=0.119, abs_rel=0.278, angle=40.40]
Epoch 31: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.540, GN=0.425, vL=2.570, mIoU=0.119, miou=0.119, abs_rel=0.278, angle=40.40]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 31: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.530, GN=0.425, vL=2.550, mIoU=0.126, miou=0.126, abs_rel=0.292, angle=39.90]
Epoch 32: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.530, GN=0.467, vL=2.550, mIoU=0.126, miou=0.126, abs_rel=0.292, angle=39.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 32: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.510, GN=0.467, vL=2.570, mIoU=0.130, miou=0.130, abs_rel=0.288, angle=40.60]
Epoch 33: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.510, GN=0.754, vL=2.570, mIoU=0.130, miou=0.130, abs_rel=0.288, angle=40.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.12it/s]
Epoch 33: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.510, GN=0.754, vL=2.560, mIoU=0.126, miou=0.126, abs_rel=0.318, angle=39.90]
Epoch 34: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.510, GN=0.310, vL=2.560, mIoU=0.126, miou=0.126, abs_rel=0.318, angle=39.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 08:42:37,396][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 34: Planner HF=0.3840, Expert HF=0.2922 (Divorce Index: 0.76x)
[2026-04-24 08:42:37,396][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.292) <= Planner HF ratio (0.384). ALB spectral separation not achieved. Check expert branch init.

Epoch 34: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.510, GN=0.310, vL=2.590, mIoU=0.130, miou=0.130, abs_rel=0.330, angle=39.90]
Epoch 35: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.510, GN=0.146, vL=2.590, mIoU=0.130, miou=0.130, abs_rel=0.330, angle=39.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 35: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.520, GN=0.146, vL=2.510, mIoU=0.134, miou=0.134, abs_rel=0.269, angle=39.80]
Epoch 36: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.520, GN=0.318, vL=2.510, mIoU=0.134, miou=0.134, abs_rel=0.269, angle=39.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 36: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.490, GN=0.318, vL=2.480, mIoU=0.135, miou=0.135, abs_rel=0.286, angle=39.20]
Epoch 37: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.490, GN=0.265, vL=2.480, mIoU=0.135, miou=0.135, abs_rel=0.286, angle=39.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 37: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.490, GN=0.265, vL=2.450, mIoU=0.140, miou=0.140, abs_rel=0.273, angle=39.30]
Epoch 38: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.490, GN=0.500, vL=2.450, mIoU=0.140, miou=0.140, abs_rel=0.273, angle=39.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 38: 100% 99/99 [02:40<00:00,  1.63s/it, L=2.490, GN=0.500, vL=2.480, mIoU=0.142, miou=0.142, abs_rel=0.267, angle=39.60]
Epoch 39: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.490, GN=0.237, vL=2.480, mIoU=0.142, miou=0.142, abs_rel=0.267, angle=39.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s][2026-04-24 08:55:57,739][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 39: Planner HF=0.3582, Expert HF=0.2587 (Divorce Index: 0.72x)
[2026-04-24 08:55:57,740][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.259) <= Planner HF ratio (0.358). ALB spectral separation not achieved. Check expert branch init.

Epoch 39: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.460, GN=0.237, vL=2.460, mIoU=0.140, miou=0.140, abs_rel=0.300, angle=39.20]
Epoch 40: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.460, GN=0.166, vL=2.460, mIoU=0.140, miou=0.140, abs_rel=0.300, angle=39.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.12it/s]
Epoch 40: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.450, GN=0.166, vL=2.470, mIoU=0.130, miou=0.130, abs_rel=0.264, angle=38.90]
Epoch 41: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.450, GN=0.163, vL=2.470, mIoU=0.130, miou=0.130, abs_rel=0.264, angle=38.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 41: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.450, GN=0.163, vL=2.450, mIoU=0.147, miou=0.147, abs_rel=0.258, angle=39.40]
Epoch 42: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.450, GN=0.408, vL=2.450, mIoU=0.147, miou=0.147, abs_rel=0.258, angle=39.40]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 42: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.430, GN=0.408, vL=2.440, mIoU=0.145, miou=0.145, abs_rel=0.266, angle=39.20]
Epoch 43: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.430, GN=0.562, vL=2.440, mIoU=0.145, miou=0.145, abs_rel=0.266, angle=39.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:20,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 43: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.420, GN=0.562, vL=2.440, mIoU=0.149, miou=0.149, abs_rel=0.281, angle=39.30]
Epoch 44: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.420, GN=0.337, vL=2.440, mIoU=0.149, miou=0.149, abs_rel=0.281, angle=39.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.10it/s][2026-04-24 09:09:17,918][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 44: Planner HF=0.3515, Expert HF=0.2581 (Divorce Index: 0.73x)
[2026-04-24 09:09:17,918][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.258) <= Planner HF ratio (0.351). ALB spectral separation not achieved. Check expert branch init.

Epoch 44: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.410, GN=0.337, vL=2.410, mIoU=0.156, miou=0.156, abs_rel=0.257, angle=39.10]
Epoch 45: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.410, GN=0.833, vL=2.410, mIoU=0.156, miou=0.156, abs_rel=0.257, angle=39.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.15it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.14it/s]
Epoch 45: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.400, GN=0.833, vL=2.430, mIoU=0.151, miou=0.151, abs_rel=0.268, angle=39.10]
Epoch 46: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.400, GN=0.372, vL=2.430, mIoU=0.151, miou=0.151, abs_rel=0.268, angle=39.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 46: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.410, GN=0.372, vL=2.400, mIoU=0.155, miou=0.155, abs_rel=0.269, angle=38.90]
[GradHealth] Step 9350: ratio 9.94e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 47: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.410, GN=0.430, vL=2.400, mIoU=0.155, miou=0.155, abs_rel=0.269, angle=38.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.14it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 47: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.380, GN=0.430, vL=2.380, mIoU=0.159, miou=0.159, abs_rel=0.262, angle=39.10]
Epoch 48: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.380, GN=0.463, vL=2.380, mIoU=0.159, miou=0.159, abs_rel=0.262, angle=39.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.10it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.11it/s]
Epoch 48: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.390, GN=0.463, vL=2.400, mIoU=0.159, miou=0.159, abs_rel=0.277, angle=39.20]
Epoch 49: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.390, GN=0.718, vL=2.400, mIoU=0.159, miou=0.159, abs_rel=0.277, angle=39.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s][2026-04-24 09:22:36,056][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 49: Planner HF=0.3505, Expert HF=0.2522 (Divorce Index: 0.72x)
[2026-04-24 09:22:36,056][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.252) <= Planner HF ratio (0.350). ALB spectral separation not achieved. Check expert branch init.

Epoch 49: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.380, GN=0.718, vL=2.450, mIoU=0.157, miou=0.157, abs_rel=0.267, angle=39.30]
Epoch 50: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.380, GN=0.688, vL=2.450, mIoU=0.157, miou=0.157, abs_rel=0.267, angle=39.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 50: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.350, GN=0.688, vL=2.380, mIoU=0.168, miou=0.168, abs_rel=0.259, angle=38.70]
[GradHealth] Step 10250: ratio 7.52e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 51: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.350, GN=0.162, vL=2.380, mIoU=0.168, miou=0.168, abs_rel=0.259, angle=38.70]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 51: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.360, GN=0.162, vL=2.340, mIoU=0.169, miou=0.169, abs_rel=0.266, angle=38.60]
[GradHealth] Step 10450: ratio 7.57e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 52: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.360, GN=0.148, vL=2.340, mIoU=0.169, miou=0.169, abs_rel=0.266, angle=38.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:19<00:59,  1.04it/s]
Validation DataLoader 0:  49% 40/82 [00:37<00:38,  1.08it/s]
Validation DataLoader 0:  73% 60/82 [00:55<00:20,  1.07it/s]
Validation DataLoader 0:  98% 80/82 [01:13<00:01,  1.09it/s]
Validation DataLoader 0: 100% 82/82 [01:15<00:00,  1.09it/s]
Epoch 52: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.370, GN=0.148, vL=2.380, mIoU=0.162, miou=0.162, abs_rel=0.274, angle=38.30]
Epoch 53: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.370, GN=0.741, vL=2.380, mIoU=0.162, miou=0.162, abs_rel=0.274, angle=38.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.11it/s]
Epoch 53: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.350, GN=0.741, vL=2.350, mIoU=0.177, miou=0.177, abs_rel=0.268, angle=38.50]
[GradHealth] Step 10800: ratio 9.34e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 54: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.350, GN=0.302, vL=2.350, mIoU=0.177, miou=0.177, abs_rel=0.268, angle=38.50]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 09:35:58,862][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 54: Planner HF=0.3409, Expert HF=0.2478 (Divorce Index: 0.73x)
[2026-04-24 09:35:58,862][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.248) <= Planner HF ratio (0.341). ALB spectral separation not achieved. Check expert branch init.

Epoch 54: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.350, GN=0.302, vL=2.330, mIoU=0.174, miou=0.174, abs_rel=0.254, angle=38.60]
[GradHealth] Step 10900: ratio 8.58e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 55: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.350, GN=0.590, vL=2.330, mIoU=0.174, miou=0.174, abs_rel=0.254, angle=38.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.10it/s]
Epoch 55: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.310, GN=0.590, vL=2.320, mIoU=0.184, miou=0.184, abs_rel=0.262, angle=38.60]
[GradHealth] Step 11100: ratio 8.92e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 56: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.310, GN=0.752, vL=2.320, mIoU=0.184, miou=0.184, abs_rel=0.262, angle=38.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.13it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 56: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.320, GN=0.752, vL=2.330, mIoU=0.175, miou=0.175, abs_rel=0.258, angle=38.40]
[GradHealth] Step 11450: ratio 8.67e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 57: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.320, GN=0.235, vL=2.330, mIoU=0.175, miou=0.175, abs_rel=0.258, angle=38.40]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 57: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.310, GN=0.235, vL=2.310, mIoU=0.187, miou=0.187, abs_rel=0.255, angle=38.60]
[GradHealth] Step 11600: ratio 9.65e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 58: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.310, GN=0.387, vL=2.310, mIoU=0.187, miou=0.187, abs_rel=0.255, angle=38.60]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 58: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.280, GN=0.387, vL=2.330, mIoU=0.175, miou=0.175, abs_rel=0.277, angle=38.30]
[GradHealth] Step 11750: ratio 8.67e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 59: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.280, GN=0.320, vL=2.330, mIoU=0.175, miou=0.175, abs_rel=0.277, angle=38.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.07it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:55<00:20,  1.09it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.10it/s][2026-04-24 09:49:22,215][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 59: Planner HF=0.3420, Expert HF=0.2463 (Divorce Index: 0.72x)
[2026-04-24 09:49:22,215][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.246) <= Planner HF ratio (0.342). ALB spectral separation not achieved. Check expert branch init.

Epoch 59: 100% 99/99 [02:41<00:00,  1.64s/it, L=2.300, GN=0.320, vL=2.310, mIoU=0.189, miou=0.189, abs_rel=0.266, angle=38.40]
[GradHealth] Step 11900: ratio 7.13e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 60: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.300, GN=0.789, vL=2.310, mIoU=0.189, miou=0.189, abs_rel=0.266, angle=38.40]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 60: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.280, GN=0.789, vL=2.290, mIoU=0.189, miou=0.189, abs_rel=0.257, angle=38.70]
[GradHealth] Step 12150: ratio 7.47e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 61: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.280, GN=0.905, vL=2.290, mIoU=0.189, miou=0.189, abs_rel=0.257, angle=38.70]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:55<00:20,  1.09it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.11it/s]
Epoch 61: 100% 99/99 [02:40<00:00,  1.63s/it, L=2.280, GN=0.905, vL=2.280, mIoU=0.201, miou=0.201, abs_rel=0.253, angle=38.30]
[GradHealth] Step 12300: ratio 7.46e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 62: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.280, GN=0.223, vL=2.280, mIoU=0.201, miou=0.201, abs_rel=0.253, angle=38.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 62: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.260, GN=0.223, vL=2.270, mIoU=0.195, miou=0.195, abs_rel=0.253, angle=38.50]
[GradHealth] Step 12500: ratio 9.07e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 63: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.260, GN=0.322, vL=2.270, mIoU=0.195, miou=0.195, abs_rel=0.253, angle=38.50]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.14it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 63: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.260, GN=0.322, vL=2.290, mIoU=0.198, miou=0.198, abs_rel=0.258, angle=38.30]
[GradHealth] Step 12700: ratio 8.10e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 64: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.260, GN=0.810, vL=2.290, mIoU=0.198, miou=0.198, abs_rel=0.258, angle=38.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s][2026-04-24 10:02:43,131][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 64: Planner HF=0.3390, Expert HF=0.2406 (Divorce Index: 0.71x)
[2026-04-24 10:02:43,132][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.241) <= Planner HF ratio (0.339). ALB spectral separation not achieved. Check expert branch init.

Epoch 64: 100% 99/99 [02:40<00:00,  1.63s/it, L=2.240, GN=0.810, vL=2.290, mIoU=0.200, miou=0.200, abs_rel=0.252, angle=38.20]
[GradHealth] Step 12900: ratio 7.30e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 65: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.240, GN=0.256, vL=2.290, mIoU=0.200, miou=0.200, abs_rel=0.252, angle=38.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:36,  1.14it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 65: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.250, GN=0.256, vL=2.280, mIoU=0.205, miou=0.205, abs_rel=0.250, angle=38.10]
[GradHealth] Step 13150: ratio 8.83e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 66: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.250, GN=0.739, vL=2.280, mIoU=0.205, miou=0.205, abs_rel=0.250, angle=38.10]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:58,  1.07it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:13<00:01,  1.09it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.10it/s]
Epoch 66: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.230, GN=0.739, vL=2.280, mIoU=0.201, miou=0.201, abs_rel=0.257, angle=38.20]
[GradHealth] Step 13350: ratio 7.78e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 67: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.230, GN=0.906, vL=2.280, mIoU=0.201, miou=0.201, abs_rel=0.257, angle=38.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.13it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 67: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.220, GN=0.906, vL=2.270, mIoU=0.209, miou=0.209, abs_rel=0.257, angle=38.00]
[GradHealth] Step 13500: ratio 5.80e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 68: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.220, GN=0.490, vL=2.270, mIoU=0.209, miou=0.209, abs_rel=0.257, angle=38.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 68: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.220, GN=0.490, vL=2.260, mIoU=0.206, miou=0.206, abs_rel=0.268, angle=38.00]
[GradHealth] Step 13700: ratio 5.56e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 69: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.220, GN=0.806, vL=2.260, mIoU=0.206, miou=0.206, abs_rel=0.268, angle=38.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 10:16:04,383][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 69: Planner HF=0.3414, Expert HF=0.2396 (Divorce Index: 0.70x)
[2026-04-24 10:16:04,383][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.240) <= Planner HF ratio (0.341). ALB spectral separation not achieved. Check expert branch init.

Epoch 69: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.230, GN=0.806, vL=2.260, mIoU=0.204, miou=0.204, abs_rel=0.253, angle=38.30]
[GradHealth] Step 13900: ratio 4.95e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 70: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.230, GN=0.398, vL=2.260, mIoU=0.204, miou=0.204, abs_rel=0.253, angle=38.30]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.10it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.12it/s]
Epoch 70: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.200, GN=0.398, vL=2.240, mIoU=0.202, miou=0.202, abs_rel=0.263, angle=37.90]
[GradHealth] Step 14100: ratio 5.50e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 71: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.200, GN=0.493, vL=2.240, mIoU=0.202, miou=0.202, abs_rel=0.263, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.10it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.11it/s]
Epoch 71: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.220, GN=0.493, vL=2.220, mIoU=0.211, miou=0.211, abs_rel=0.243, angle=37.90]
[GradHealth] Step 14300: ratio 4.67e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 72: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.220, GN=0.552, vL=2.220, mIoU=0.211, miou=0.211, abs_rel=0.243, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 72: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.200, GN=0.552, vL=2.230, mIoU=0.210, miou=0.210, abs_rel=0.242, angle=37.80]
[GradHealth] Step 14500: ratio 5.70e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 73: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.200, GN=0.645, vL=2.230, mIoU=0.210, miou=0.210, abs_rel=0.242, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 73: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.210, GN=0.645, vL=2.230, mIoU=0.207, miou=0.207, abs_rel=0.248, angle=38.20]
[GradHealth] Step 14700: ratio 3.20e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 74: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.210, GN=0.745, vL=2.230, mIoU=0.207, miou=0.207, abs_rel=0.248, angle=38.20]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s][2026-04-24 10:29:24,855][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 74: Planner HF=0.3362, Expert HF=0.2381 (Divorce Index: 0.71x)
[2026-04-24 10:29:24,856][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.238) <= Planner HF ratio (0.336). ALB spectral separation not achieved. Check expert branch init.

Epoch 74: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.210, GN=0.745, vL=2.230, mIoU=0.218, miou=0.218, abs_rel=0.254, angle=38.00]
[GradHealth] Step 14900: ratio 5.02e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 75: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.210, GN=0.654, vL=2.230, mIoU=0.218, miou=0.218, abs_rel=0.254, angle=38.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:20,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 75: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.190, GN=0.654, vL=2.220, mIoU=0.217, miou=0.217, abs_rel=0.250, angle=37.90]
[GradHealth] Step 15050: ratio 3.34e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 76: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.190, GN=0.416, vL=2.220, mIoU=0.217, miou=0.217, abs_rel=0.250, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 76: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.210, GN=0.416, vL=2.230, mIoU=0.215, miou=0.215, abs_rel=0.244, angle=37.90]
[GradHealth] Step 15250: ratio 2.89e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 77: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.210, GN=0.521, vL=2.230, mIoU=0.215, miou=0.215, abs_rel=0.244, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.08it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:20,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 77: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.190, GN=0.521, vL=2.230, mIoU=0.214, miou=0.214, abs_rel=0.247, angle=38.00]
[GradHealth] Step 15450: ratio 3.80e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 78: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.190, GN=0.494, vL=2.230, mIoU=0.214, miou=0.214, abs_rel=0.247, angle=38.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.09it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.10it/s]
Validation DataLoader 0:  98% 80/82 [01:13<00:01,  1.09it/s]
Validation DataLoader 0: 100% 82/82 [01:14<00:00,  1.09it/s]
Epoch 78: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.180, GN=0.494, vL=2.220, mIoU=0.215, miou=0.215, abs_rel=0.249, angle=37.90]
[GradHealth] Step 15650: ratio 2.28e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 79: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.180, GN=0.899, vL=2.220, mIoU=0.215, miou=0.215, abs_rel=0.249, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 10:42:47,072][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 79: Planner HF=0.3268, Expert HF=0.2353 (Divorce Index: 0.72x)
[2026-04-24 10:42:47,073][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.235) <= Planner HF ratio (0.327). ALB spectral separation not achieved. Check expert branch init.

Epoch 79: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.190, GN=0.899, vL=2.210, mIoU=0.216, miou=0.216, abs_rel=0.241, angle=37.80]
[GradHealth] Step 15850: ratio 3.54e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 80: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.190, GN=0.665, vL=2.210, mIoU=0.216, miou=0.216, abs_rel=0.241, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 80: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.170, GN=0.665, vL=2.210, mIoU=0.217, miou=0.217, abs_rel=0.244, angle=37.80]
[GradHealth] Step 16050: ratio 1.66e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 81: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.170, GN=0.169, vL=2.210, mIoU=0.217, miou=0.217, abs_rel=0.244, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.14it/s]
Epoch 81: 100% 99/99 [02:38<00:00,  1.60s/it, L=2.160, GN=0.169, vL=2.220, mIoU=0.219, miou=0.219, abs_rel=0.242, angle=37.90]
[GradHealth] Step 16250: ratio 1.56e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 82: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.160, GN=0.396, vL=2.220, mIoU=0.219, miou=0.219, abs_rel=0.242, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:57,  1.07it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 82: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.170, GN=0.396, vL=2.210, mIoU=0.219, miou=0.219, abs_rel=0.247, angle=37.80]
[GradHealth] Step 16450: ratio 2.54e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 83: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.170, GN=0.721, vL=2.210, mIoU=0.219, miou=0.219, abs_rel=0.247, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.15it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.14it/s]
Epoch 83: 100% 99/99 [02:38<00:00,  1.60s/it, L=2.170, GN=0.721, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.246, angle=37.80]
[GradHealth] Step 16650: ratio 1.72e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 84: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.170, GN=0.950, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.246, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:58,  1.07it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s][2026-04-24 10:56:05,055][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 84: Planner HF=0.3303, Expert HF=0.2370 (Divorce Index: 0.72x)
[2026-04-24 10:56:05,055][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.237) <= Planner HF ratio (0.330). ALB spectral separation not achieved. Check expert branch init.

Epoch 84: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.180, GN=0.950, vL=2.200, mIoU=0.218, miou=0.218, abs_rel=0.246, angle=37.80]
[GradHealth] Step 16850: ratio 1.56e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 85: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.180, GN=0.596, vL=2.200, mIoU=0.218, miou=0.218, abs_rel=0.246, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 85: 100% 99/99 [02:39<00:00,  1.62s/it, L=2.170, GN=0.596, vL=2.210, mIoU=0.223, miou=0.223, abs_rel=0.242, angle=38.00]
[GradHealth] Step 17050: ratio 8.54e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 86: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.170, GN=0.346, vL=2.210, mIoU=0.223, miou=0.223, abs_rel=0.242, angle=38.00]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.09it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.12it/s]
Epoch 86: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.150, GN=0.346, vL=2.200, mIoU=0.223, miou=0.223, abs_rel=0.242, angle=37.80]
[GradHealth] Step 17250: ratio 1.09e-07 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 87: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.150, GN=0.841, vL=2.200, mIoU=0.223, miou=0.223, abs_rel=0.242, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:53,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 87: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.140, GN=0.841, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.239, angle=37.80]
[GradHealth] Step 17450: ratio 7.05e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 88: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.140, GN=0.334, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.239, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:55,  1.11it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 88: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.150, GN=0.334, vL=2.200, mIoU=0.223, miou=0.223, abs_rel=0.245, angle=37.90]
[GradHealth] Step 17650: ratio 6.08e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 89: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.150, GN=0.901, vL=2.200, mIoU=0.223, miou=0.223, abs_rel=0.245, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 11:09:24,868][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 89: Planner HF=0.3276, Expert HF=0.2347 (Divorce Index: 0.72x)
[2026-04-24 11:09:24,869][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.235) <= Planner HF ratio (0.328). ALB spectral separation not achieved. Check expert branch init.

Epoch 89: 100% 99/99 [02:40<00:00,  1.63s/it, L=2.160, GN=0.901, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.240, angle=37.80]
[GradHealth] Step 17850: ratio 9.20e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 90: 100% 99/99 [01:20<00:00,  1.23it/s, L=2.160, GN=0.909, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.240, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.14it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:36,  1.14it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 90: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.170, GN=0.909, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.251, angle=37.70]
[GradHealth] Step 18050: ratio 7.11e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 91: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.170, GN=0.779, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.251, angle=37.70]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 91: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.150, GN=0.779, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.241, angle=37.80]
[GradHealth] Step 18250: ratio 4.82e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 92: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.150, GN=0.472, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.241, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 92: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.150, GN=0.472, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.248, angle=37.70]
[GradHealth] Step 18450: ratio 3.37e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 93: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.150, GN=0.260, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.248, angle=37.70]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:38,  1.10it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.11it/s]
Epoch 93: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.140, GN=0.260, vL=2.200, mIoU=0.224, miou=0.224, abs_rel=0.240, angle=37.90]
[GradHealth] Step 18650: ratio 3.42e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 94: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.140, GN=0.427, vL=2.200, mIoU=0.224, miou=0.224, abs_rel=0.240, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.13it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.13it/s]
Validation DataLoader 0:  73% 60/82 [00:54<00:19,  1.11it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.12it/s][2026-04-24 11:22:45,681][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 94: Planner HF=0.3304, Expert HF=0.2370 (Divorce Index: 0.72x)
[2026-04-24 11:22:45,681][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.237) <= Planner HF ratio (0.330). ALB spectral separation not achieved. Check expert branch init.

Epoch 94: 100% 99/99 [02:41<00:00,  1.63s/it, L=2.140, GN=0.427, vL=2.190, mIoU=0.223, miou=0.223, abs_rel=0.247, angle=37.80]
[GradHealth] Step 18850: ratio 2.02e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 95: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.140, GN=0.594, vL=2.190, mIoU=0.223, miou=0.223, abs_rel=0.247, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:18<00:56,  1.10it/s]
Validation DataLoader 0:  49% 40/82 [00:36<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:12<00:01,  1.11it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 95: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.140, GN=0.594, vL=2.200, mIoU=0.223, miou=0.223, abs_rel=0.247, angle=37.90]
[GradHealth] Step 19050: ratio 1.16e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 96: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.140, GN=0.735, vL=2.200, mIoU=0.223, miou=0.223, abs_rel=0.247, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.15it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.14it/s]
Epoch 96: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.150, GN=0.735, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.241, angle=37.80]
[GradHealth] Step 19250: ratio 1.13e-08 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 97: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.150, GN=0.246, vL=2.200, mIoU=0.220, miou=0.220, abs_rel=0.241, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:55,  1.12it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.12it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s]
Epoch 97: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.140, GN=0.246, vL=2.200, mIoU=0.224, miou=0.224, abs_rel=0.241, angle=37.90]
[GradHealth] Step 19450: ratio 5.89e-09 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 98: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.140, GN=0.195, vL=2.200, mIoU=0.224, miou=0.224, abs_rel=0.241, angle=37.90]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.15it/s]
Validation DataLoader 0:  49% 40/82 [00:34<00:36,  1.15it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.12it/s]
Validation DataLoader 0:  98% 80/82 [01:10<00:01,  1.13it/s]
Validation DataLoader 0: 100% 82/82 [01:12<00:00,  1.13it/s]
Epoch 98: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.140, GN=0.195, vL=2.200, mIoU=0.221, miou=0.221, abs_rel=0.240, angle=37.80]
[GradHealth] Step 19650: ratio 6.92e-09 is very small — possible vanishing gradient. (Silencing further warnings this epoch)
Epoch 99: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.140, GN=0.555, vL=2.200, mIoU=0.221, miou=0.221, abs_rel=0.240, angle=37.80]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/82 [00:00<?, ?it/s]
Validation DataLoader 0:  24% 20/82 [00:17<00:54,  1.13it/s]
Validation DataLoader 0:  49% 40/82 [00:35<00:37,  1.11it/s]
Validation DataLoader 0:  73% 60/82 [00:53<00:19,  1.13it/s]
Validation DataLoader 0:  98% 80/82 [01:11<00:01,  1.12it/s]
Validation DataLoader 0: 100% 82/82 [01:13<00:00,  1.12it/s][2026-04-24 11:36:03,887][spectra.callbacks][INFO] - [Spectral-Audit] Epoch 99: Planner HF=0.3310, Expert HF=0.2386 (Divorce Index: 0.72x)
[2026-04-24 11:36:03,887][spectra.callbacks][WARNING] - [Spectral-Audit] DEGRADATION: Expert HF ratio (0.239) <= Planner HF ratio (0.331). ALB spectral separation not achieved. Check expert branch init.

Epoch 99: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.150, GN=0.555, vL=2.200, mIoU=0.222, miou=0.222, abs_rel=0.242, angle=37.90]
`Trainer.fit` stopped: `max_epochs=100` reached.
[2026-04-24 11:36:10,303][spectra.runner][INFO] - [Mission-Control] Mission Accomplished. [SUCCESS]

[Experiment 'bpgs_alb_nyuv2' Finished]
  Status:  completed
  End:     2026-04-24 11:36:15+00:00
  Elapsed: 16031.320s