------------------------------------------------------------
[2026-03-05 22:59:23] Launching: bpgs
------------------------------------------------------------
Seed set to 42
[2026-03-05 22:59:31,775][spectra.runner][INFO] - [Mission-Control] Workspace: /content/SPECTRA/outputs/2026-03-05/22-59-31
[2026-03-05 22:59:31,779][spectra.runner][INFO] - [Mission-Control] Config:
dataset_name: synthetic
module:
  _target_: spectra.modules.synthetic.SyntheticSPECTRAModule
  engine:
    _target_: spectra.engine.optimizers.BPGSEngine
model:
  backbone: shared_trunk
  input_dim: 20
  d_model: 128
  hidden_layers: 3
  n_heads: 8
  dropout: 0.1
tasks:
- name: mse_high
  type: regression
  loss: mse
  manifold: planner
- name: mse_med
  type: regression
  loss: mse
  manifold: planner
- name: mse_low
  type: regression
  loss: mse
  manifold: planner
- name: bce_0
  type: classification
  loss: bce
  manifold: expert
- name: bce_1
  type: classification
  loss: bce
  manifold: expert
- name: bce_2
  type: classification
  loss: bce
  manifold: expert
- name: mse_aux
  type: regression
  loss: mse
  manifold: planner
train:
  save_ckpt: false
  epochs: 80
  batch_size: 256
  lr: 0.001
  min_lr: 1.0e-05
  warmup_steps: 200
  weight_decay: 0.0001
  grad_clip: 2.0
  precision: 16-mixed
  num_workers: 2
  log_every_n_steps: 50
  early_stop: true
  early_stop_patience: 20
  early_stop_min_delta: 0.0001
data:
  n_train: 20000
  n_val: 5000
method_name: ${method.name}
use_alb: false
method:
  name: bpgs
  use_sigmoid: true
  use_autocal: true
  ema_decay: 0.99
run_name: ${method_name}_${dataset_name}_s${seed}
seed: 42
logging:
  use_wandb: false
  wandb_project: spectra-mtl
  wandb_mode: offline
output_dir: ./outputs/${run_name}

[2026-03-05 22:59:31,921][spectra.preflight][INFO] - [PreFlight] All systems nominal. GO for training.
/content/SPECTRA/spectra/baselines/__init__.py:37: UserWarning: tau=0.21714724095162594 is very small (< 1 step). beta = 1 - exp(-1/tau) = 0.9900, meaning the EMA reacts almost instantly to each batch. Consider tau >= 10.
  return BPGS(
[2026-03-05 22:59:31,960][spectra.runner][INFO] - [Logging] CSV Logger initialized in artifact shell: /content/SPECTRA/outputs/2026-03-05/22-59-31/csv_logs/bpgs_20260305_225931
[2026-03-05 22:59:31,960][spectra.runner][INFO] - [Mission-Control] Manual optimization active (method=bpgs). Automatic PL gradient clipping disabled — engine manages clipping internally.
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
[2026-03-05 22:59:31,996][spectra.runner][INFO] - [Mission-Control] All systems GO. Method=bpgs, Epochs=80, Tasks=['mse_high', 'mse_med', 'mse_low', 'bce_0', 'bce_1', 'bce_2', 'mse_aux']
[2026-03-05 22:59:32,340][spectra.datamodule][INFO] - [DataModule] Setup complete: synthetic (Train=20000, Val=5000)
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/model_summary/model_summary.py:242: Precision 16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name           ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model          │ DynamicWrapper │  457 K │ train │     0 │
│ 1 │ backbone       │ SharedTrunk    │  399 K │ train │     0 │
│ 2 │ heads          │ ModuleDict     │ 58.2 K │ train │     0 │
│ 3 │ weighter       │ BPGS           │      7 │ train │     0 │
│ 4 │ task_weights   │ ParameterDict  │      7 │ train │     0 │
│ 5 │ task_losses    │ ModuleDict     │      0 │ train │     0 │
│ 6 │ target_scalers │ ModuleDict     │      0 │ train │     0 │
└───┴────────────────┴────────────────┴────────┴───────┴───────┘
Trainable params: 457 K                                                         
Non-trainable params: 7                                                         
Total params: 457 K                                                             
Total estimated model params size (MB): 1                                       
Modules in train mode: 83                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Training   0% 0/? [00:00<?, ?it/s]         
Epoch 0: 100% 78/78 [00:05<00:00, 15.58it/s, L=1.740]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.85it/s]
Epoch 0: 100% 78/78 [00:05<00:00, 14.36it/s, L=3.690, vL=1.850]
Epoch 1: 100% 78/78 [00:03<00:00, 19.77it/s, L=3.690, vL=1.850]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.63it/s]
Epoch 1: 100% 78/78 [00:04<00:00, 17.87it/s, L=1.430, vL=1.120]
Epoch 2: 100% 78/78 [00:03<00:00, 19.79it/s, L=1.430, vL=1.120]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 30.44it/s]
Epoch 2: 100% 78/78 [00:04<00:00, 16.77it/s, L=1.090, vL=1.020]
Epoch 3: 100% 78/78 [00:04<00:00, 16.42it/s, L=1.090, vL=1.020]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 49.50it/s]
Epoch 3: 100% 78/78 [00:05<00:00, 15.03it/s, L=0.970, vL=0.916]
Epoch 4: 100% 78/78 [00:04<00:00, 19.39it/s, L=0.970, vL=0.916]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 49.90it/s]
Epoch 4: 100% 78/78 [00:04<00:00, 17.49it/s, L=0.904, vL=0.885]
Epoch 5: 100% 78/78 [00:04<00:00, 16.94it/s, L=0.904, vL=0.885]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 35.24it/s]
Epoch 5: 100% 78/78 [00:05<00:00, 14.92it/s, L=0.860, vL=0.859]
Epoch 6: 100% 78/78 [00:03<00:00, 19.55it/s, L=0.860, vL=0.859]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 47.89it/s]
Epoch 6: 100% 78/78 [00:04<00:00, 17.54it/s, L=0.825, vL=0.840]
Epoch 7: 100% 78/78 [00:04<00:00, 17.90it/s, L=0.825, vL=0.840]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 48.68it/s]
Epoch 7: 100% 78/78 [00:04<00:00, 16.25it/s, L=0.798, vL=0.833]
Epoch 8: 100% 78/78 [00:04<00:00, 16.31it/s, L=0.798, vL=0.833]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.19it/s]
Epoch 8: 100% 78/78 [00:05<00:00, 14.98it/s, L=0.774, vL=0.821]
Epoch 9: 100% 78/78 [00:03<00:00, 20.05it/s, L=0.774, vL=0.821]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 50.77it/s]
Epoch 9: 100% 78/78 [00:04<00:00, 18.07it/s, L=0.752, vL=0.825]
Epoch 10: 100% 78/78 [00:03<00:00, 19.93it/s, L=0.752, vL=0.825]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 36.52it/s]
Epoch 10: 100% 78/78 [00:04<00:00, 17.28it/s, L=0.742, vL=0.811]
Epoch 11: 100% 78/78 [00:04<00:00, 16.81it/s, L=0.742, vL=0.811]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 52.24it/s]
Epoch 11: 100% 78/78 [00:05<00:00, 15.41it/s, L=0.727, vL=0.828]
Epoch 12: 100% 78/78 [00:03<00:00, 19.63it/s, L=0.727, vL=0.828]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 50.61it/s]
Epoch 12: 100% 78/78 [00:04<00:00, 17.70it/s, L=0.714, vL=0.798]
Epoch 13: 100% 78/78 [00:05<00:00, 15.35it/s, L=0.714, vL=0.798]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.20it/s]
Epoch 13: 100% 78/78 [00:05<00:00, 14.16it/s, L=0.704, vL=0.812]
Epoch 14: 100% 78/78 [00:04<00:00, 19.46it/s, L=0.704, vL=0.812]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 49.16it/s]
Epoch 14: 100% 78/78 [00:04<00:00, 17.53it/s, L=0.701, vL=0.788]
Epoch 15: 100% 78/78 [00:03<00:00, 19.68it/s, L=0.701, vL=0.788]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 52.53it/s]
Epoch 15: 100% 78/78 [00:04<00:00, 17.80it/s, L=0.686, vL=0.807]
Epoch 16: 100% 78/78 [00:04<00:00, 16.19it/s, L=0.686, vL=0.807]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.63it/s]
Epoch 16: 100% 78/78 [00:05<00:00, 14.88it/s, L=0.676, vL=0.792]
Epoch 17: 100% 78/78 [00:03<00:00, 19.99it/s, L=0.676, vL=0.792]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.17it/s]
Epoch 17: 100% 78/78 [00:04<00:00, 18.02it/s, L=0.668, vL=0.783]
Epoch 18: 100% 78/78 [00:04<00:00, 19.34it/s, L=0.668, vL=0.783]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 36.25it/s]
Epoch 18: 100% 78/78 [00:04<00:00, 16.80it/s, L=0.659, vL=0.780]
Epoch 19: 100% 78/78 [00:04<00:00, 17.75it/s, L=0.659, vL=0.780]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 50.67it/s]
Epoch 19: 100% 78/78 [00:04<00:00, 16.18it/s, L=0.652, vL=0.792]
Epoch 20: 100% 78/78 [00:04<00:00, 18.66it/s, L=0.652, vL=0.792]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 45.91it/s]
Epoch 20: 100% 78/78 [00:04<00:00, 16.76it/s, L=0.650, vL=0.791]
Epoch 21: 100% 78/78 [00:04<00:00, 15.94it/s, L=0.650, vL=0.791]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 47.19it/s]
Epoch 21: 100% 78/78 [00:05<00:00, 14.47it/s, L=0.638, vL=0.824]
Epoch 22: 100% 78/78 [00:04<00:00, 19.40it/s, L=0.638, vL=0.824]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 52.44it/s]
Epoch 22: 100% 78/78 [00:04<00:00, 17.57it/s, L=0.637, vL=0.787]
Epoch 23: 100% 78/78 [00:03<00:00, 19.52it/s, L=0.637, vL=0.787]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 53.07it/s]
Epoch 23: 100% 78/78 [00:04<00:00, 17.69it/s, L=0.630, vL=0.802]
Epoch 24: 100% 78/78 [00:04<00:00, 16.56it/s, L=0.630, vL=0.802]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.08it/s]
Epoch 24: 100% 78/78 [00:05<00:00, 15.19it/s, L=0.617, vL=0.799]
Epoch 25: 100% 78/78 [00:04<00:00, 19.41it/s, L=0.617, vL=0.799]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 48.22it/s]
Epoch 25: 100% 78/78 [00:04<00:00, 17.46it/s, L=0.611, vL=0.797]
Epoch 26: 100% 78/78 [00:04<00:00, 18.63it/s, L=0.611, vL=0.797]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 37.10it/s]
Epoch 26: 100% 78/78 [00:04<00:00, 16.38it/s, L=0.604, vL=0.787]
Epoch 27: 100% 78/78 [00:04<00:00, 17.91it/s, L=0.604, vL=0.787]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 46.98it/s]
Epoch 27: 100% 78/78 [00:04<00:00, 16.20it/s, L=0.597, vL=0.806]
Epoch 28: 100% 78/78 [00:04<00:00, 19.44it/s, L=0.597, vL=0.806]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.84it/s]
Epoch 28: 100% 78/78 [00:04<00:00, 17.59it/s, L=0.598, vL=0.820]
Epoch 29: 100% 78/78 [00:05<00:00, 15.28it/s, L=0.598, vL=0.820]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 28.81it/s]
Epoch 29: 100% 78/78 [00:05<00:00, 13.36it/s, L=0.583, vL=0.809]
Epoch 30: 100% 78/78 [00:04<00:00, 19.15it/s, L=0.583, vL=0.809]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.31it/s]
Epoch 30: 100% 78/78 [00:04<00:00, 17.32it/s, L=0.580, vL=0.797]
Epoch 31: 100% 78/78 [00:03<00:00, 19.58it/s, L=0.580, vL=0.797]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 35.16it/s]
Epoch 31: 100% 78/78 [00:04<00:00, 17.00it/s, L=0.573, vL=0.821]
Epoch 32: 100% 78/78 [00:04<00:00, 16.45it/s, L=0.573, vL=0.821]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 50.80it/s]
Epoch 32: 100% 78/78 [00:05<00:00, 15.08it/s, L=0.565, vL=0.817]
Epoch 33: 100% 78/78 [00:03<00:00, 19.74it/s, L=0.565, vL=0.817]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 49.07it/s]
Epoch 33: 100% 78/78 [00:04<00:00, 17.74it/s, L=0.560, vL=0.819]
Epoch 34: 100% 78/78 [00:04<00:00, 16.49it/s, L=0.560, vL=0.819]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 41.16it/s]
Epoch 34: 100% 78/78 [00:05<00:00, 14.80it/s, L=0.554, vL=0.817]
Epoch 35: 100% 78/78 [00:03<00:00, 19.87it/s, L=0.554, vL=0.817]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 49.53it/s]
Epoch 35: 100% 78/78 [00:04<00:00, 17.87it/s, L=0.546, vL=0.827]
Epoch 36: 100% 78/78 [00:03<00:00, 20.02it/s, L=0.546, vL=0.827]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 51.03it/s]
Epoch 36: 100% 78/78 [00:04<00:00, 18.05it/s, L=0.541, vL=0.840]
Epoch 37: 100% 78/78 [00:04<00:00, 16.14it/s, L=0.541, vL=0.840]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 52.48it/s]
Epoch 37: 100% 78/78 [00:05<00:00, 14.87it/s, L=0.532, vL=0.834]
Epoch 38: 100% 78/78 [00:04<00:00, 18.85it/s, L=0.532, vL=0.834]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/20 [00:00<?, ?it/s]
Validation DataLoader 0: 100% 20/20 [00:00<00:00, 48.70it/s]
Epoch 38: 100% 78/78 [00:04<00:00, 16.99it/s, L=0.522, vL=0.824]
[2026-03-05 23:02:39,337][spectra.runner][INFO] - [Mission-Control] Mission Accomplished. [SUCCESS]
[2026-03-05 23:02:40] Success: bpgs completed.