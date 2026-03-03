Seed set to 42
[2026-03-02 23:16:47,880][spectra.train][INFO] - [Mission-Control] Workspace: /content/SPECTRA/outputs/2026-03-02/23-16-47
[2026-03-02 23:16:47,884][spectra.train][INFO] - [Mission-Control] Config:
run_name: ${method.name}_synthetic_s${seed}
seed: 42
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
  epochs: 500
  batch_size: 256
  lr: 0.001
  min_lr: 1.0e-05
  warmup_steps: 200
  weight_decay: 0.0001
  grad_clip: 2.0
  precision: 32
  num_workers: 2
  log_every_n_steps: 50
data:
  n_train: 10000
  n_val: 2000
logging:
  use_wandb: true
  wandb_project: spectra-mtl
  wandb_mode: offline
output_dir: ./outputs/${run_name}
method:
  name: pcgrad
  use_alb: false
dataset:
  name: clinical
  dataset_dir: datasets/sepsis_clinical_28
  hf_repo: hellxhell/sepsis-clinical-28
  sepsis_boost: 5.0
  sampler_target: outcome
  subset_pct: 1.0
  model:
    backbone: shared_trunk
    input_dim: 28
    d_model: 512
    hidden_layers: 4
    dropout: 0.2
  tasks:
  - name: outcome
    type: classification
    loss: bce
    num_classes: 1
    manifold: expert
    weight: 1.0
    pos_weight: 3.0
  - name: phase
    type: classification
    loss: cross_entropy
    num_classes: 3
    manifold: planner
    weight: 1.0
  train:
    epochs: 100
    batch_size: 64
    lr: 5.0e-05
    min_lr: 1.0e-06
    warmup_steps: 300
    weight_decay: 0.0001
    grad_clip: 1.0
    precision: 16-mixed
    log_every_n_steps: 10
    early_stop: true
    early_stop_patience: 3
    early_stop_min_delta: 0.001

[2026-03-02 23:16:48,032][spectra.train][INFO] - [PreFlight] All systems nominal. GO for training. 🚀
[2026-03-02 23:16:48,056][spectra.trainer][INFO] - [SPECTRA] Initialized: backbone=shared_trunk, method=pcgrad, ALB=False, tasks=2
[2026-03-02 23:16:48,060][spectra.trainer][INFO] - [SPECTRA] Task metrics initialized: ['outcome', 'phase']
[2026-03-02 23:16:48,063][spectra.train][INFO] - [Logging] WandB Offline Mode Engaged (Silent Research)
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id x2w1epxg.
wandb: Tracking run with wandb version 0.25.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in /content/SPECTRA/outputs/2026-03-02/23-16-47/wandb/offline-run-20260302_231648-x2w1epxg
[2026-03-02 23:16:49,814][spectra.train][INFO] - [Mission-Control] PCGrad detected (manual optimization). Automatic gradient clipping disabled — clipping handled inside training_step.
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
[2026-03-02 23:16:49,851][spectra.train][INFO] - [Mission-Control] All systems GO. Method=pcgrad, Epochs=100, Tasks=['outcome', 'phase']
[2026-03-02 23:16:50,040][APEX_Data_Frontier][INFO] - [Tier 0] Valid Local Data Found at 'datasets/sepsis_clinical_28'. System Ready.
[2026-03-02 23:16:50,041][APEX_Data_Frontier][INFO] - [TRAIN] Loading Index: datasets/sepsis_clinical_28/train_index.json
[2026-03-02 23:16:50,398][APEX_Data_Frontier][INFO] - [TRAIN] Initialized. Windows: 449,044 | Episodes: 23,739
[2026-03-02 23:16:50,399][APEX_Data_Frontier][INFO] - Augmentation Active: Noise=0.0, MaskDrop=0.0
[2026-03-02 23:16:50,400][APEX_Data_Frontier][INFO] - [VAL] Loading Index: datasets/sepsis_clinical_28/val_index.json
[2026-03-02 23:16:50,434][APEX_Data_Frontier][INFO] - [VAL] Initialized. Windows: 49,143 | Episodes: 2,540
[2026-03-02 23:16:50,434][spectra.datamodule][INFO] - [DataModule] Setup complete: clinical (Train=449044, Val=49143)
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
[2026-03-02 23:16:51,093][APEX_Data_Frontier][INFO] - [Sampler] Loading cached Sepsis Index: datasets/sepsis_clinical_28/train/train_sepsis_index_outcome_sub1.0.npy
[2026-03-02 23:16:51,095][APEX_Data_Frontier][INFO] - [Sampler] Coverage: 100% | Sepsis Detected: 13,472 | Rate: 3.00% | Boost factor: 5.0x
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/model_summary/model_summary.py:242: Precision 16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ SharedTrunk    │  2.1 M │ train │     0 │
│ 1 │ heads         │ ModuleDict     │  263 K │ train │     0 │
│ 2 │ task_losses   │ ModuleDict     │      0 │ train │     0 │
│ 3 │ _val_losses   │ ModuleDict     │      0 │ train │     0 │
│ 4 │ weighter      │ PCGradWeighter │      0 │ train │     0 │
│ 5 │ _val_metrics  │ ModuleDict     │      0 │ train │     0 │
│ 6 │ train_metrics │ ModuleDict     │      0 │ train │     0 │
└───┴───────────────┴────────────────┴────────┴───────┴───────┘
Trainable params: 2.4 M                                                         
Non-trainable params: 0                                                         
Total params: 2.4 M                                                             
Total estimated model params size (MB): 9                                       
Modules in train mode: 64                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Sanity Checking DataLoader 0: 100% 2/2 [00:00<00:00,  5.60it/s][2026-03-02 23:16:51,868][spectra.trainer][INFO] - [Val] outcome: AUC=0.0000, PRC=-0.0000, R=0.0000
[2026-03-02 23:16:51,945][spectra.trainer][INFO] - [Val] phase: ACC=0.0495, AUC=0.0000
Epoch 0: 100% 7016/7016 [05:12<00:00, 22.45it/s, GN=0.850, C=12.00, L=1.790]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/768 [00:00<?, ?it/s]
Validation DataLoader 0:   3% 20/768 [00:00<00:11, 64.20it/s]
Validation DataLoader 0:   5% 40/768 [00:00<00:11, 65.49it/s]
Validation DataLoader 0:   8% 60/768 [00:00<00:10, 65.17it/s]
Validation DataLoader 0:  10% 80/768 [00:01<00:10, 64.94it/s]
Validation DataLoader 0:  13% 100/768 [00:01<00:10, 63.94it/s]
Validation DataLoader 0:  16% 120/768 [00:01<00:10, 64.35it/s]
Validation DataLoader 0:  18% 140/768 [00:02<00:09, 64.30it/s]
Validation DataLoader 0:  21% 160/768 [00:02<00:09, 64.46it/s]
Validation DataLoader 0:  23% 180/768 [00:02<00:09, 64.16it/s]
Validation DataLoader 0:  26% 200/768 [00:03<00:08, 63.99it/s]
Validation DataLoader 0:  29% 220/768 [00:03<00:08, 63.83it/s]
Validation DataLoader 0:  31% 240/768 [00:03<00:08, 63.59it/s]
Validation DataLoader 0:  34% 260/768 [00:04<00:07, 63.61it/s]
Validation DataLoader 0:  36% 280/768 [00:04<00:07, 63.67it/s]
Validation DataLoader 0:  39% 300/768 [00:04<00:07, 63.50it/s]
Validation DataLoader 0:  42% 320/768 [00:05<00:07, 63.54it/s]
Validation DataLoader 0:  44% 340/768 [00:05<00:06, 63.68it/s]
Validation DataLoader 0:  47% 360/768 [00:05<00:06, 63.78it/s]
Validation DataLoader 0:  49% 380/768 [00:05<00:06, 63.62it/s]
Validation DataLoader 0:  52% 400/768 [00:06<00:05, 63.60it/s]
Validation DataLoader 0:  55% 420/768 [00:06<00:05, 63.57it/s]
Validation DataLoader 0:  57% 440/768 [00:06<00:05, 63.36it/s]
Validation DataLoader 0:  60% 460/768 [00:07<00:04, 63.45it/s]
Validation DataLoader 0:  62% 480/768 [00:07<00:04, 63.54it/s]
Validation DataLoader 0:  65% 500/768 [00:07<00:04, 63.50it/s]
Validation DataLoader 0:  68% 520/768 [00:08<00:03, 63.43it/s]
Validation DataLoader 0:  70% 540/768 [00:08<00:03, 63.43it/s]
Validation DataLoader 0:  73% 560/768 [00:08<00:03, 63.40it/s]
Validation DataLoader 0:  76% 580/768 [00:09<00:02, 63.35it/s]
Validation DataLoader 0:  78% 600/768 [00:09<00:02, 62.59it/s]
Validation DataLoader 0:  81% 620/768 [00:10<00:02, 61.84it/s]
Validation DataLoader 0:  83% 640/768 [00:10<00:02, 61.20it/s]
Validation DataLoader 0:  86% 660/768 [00:10<00:01, 60.65it/s]
Validation DataLoader 0:  89% 680/768 [00:11<00:01, 59.97it/s]
Validation DataLoader 0:  91% 700/768 [00:11<00:01, 59.33it/s]
Validation DataLoader 0:  94% 720/768 [00:12<00:00, 58.55it/s]
Validation DataLoader 0:  96% 740/768 [00:12<00:00, 58.12it/s]
Validation DataLoader 0:  99% 760/768 [00:13<00:00, 57.60it/s]
Validation DataLoader 0: 100% 768/768 [00:13<00:00, 57.78it/s][2026-03-02 23:22:17,853][spectra.trainer][INFO] - [Val] outcome: AUC=0.7826, PRC=0.1094, R=0.6263
[2026-03-02 23:22:17,861][spectra.trainer][INFO] - [Val] phase: ACC=0.3333, AUC=0.7786

Epoch 1: 100% 7016/7016 [05:09<00:00, 22.65it/s, GN=0.762, C=4.000, L=1.130, vL=0.533, outcome_AUC=0.783, outcome_PRC=0.109, outcome_R=0.626, AUC=0.781, phase_ACC=0.333, phase_AUC=0.779]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/768 [00:00<?, ?it/s]
Validation DataLoader 0:   3% 20/768 [00:00<00:11, 63.67it/s]
Validation DataLoader 0:   5% 40/768 [00:00<00:11, 62.70it/s]
Validation DataLoader 0:   8% 60/768 [00:00<00:11, 62.40it/s]
Validation DataLoader 0:  10% 80/768 [00:01<00:10, 63.09it/s]
Validation DataLoader 0:  13% 100/768 [00:01<00:10, 63.69it/s]
Validation DataLoader 0:  16% 120/768 [00:01<00:10, 63.52it/s]
Validation DataLoader 0:  18% 140/768 [00:02<00:09, 63.31it/s]
Validation DataLoader 0:  21% 160/768 [00:02<00:09, 63.69it/s]
Validation DataLoader 0:  23% 180/768 [00:02<00:09, 63.70it/s]
Validation DataLoader 0:  26% 200/768 [00:03<00:08, 63.34it/s]
Validation DataLoader 0:  29% 220/768 [00:03<00:08, 63.52it/s]
Validation DataLoader 0:  31% 240/768 [00:03<00:08, 62.35it/s]
Validation DataLoader 0:  34% 260/768 [00:04<00:08, 60.32it/s]
Validation DataLoader 0:  36% 280/768 [00:04<00:08, 59.20it/s]
Validation DataLoader 0:  39% 300/768 [00:05<00:08, 57.90it/s]
Validation DataLoader 0:  42% 320/768 [00:05<00:07, 56.89it/s]
Validation DataLoader 0:  44% 340/768 [00:06<00:07, 56.01it/s]
Validation DataLoader 0:  47% 360/768 [00:06<00:07, 54.91it/s]
Validation DataLoader 0:  49% 380/768 [00:07<00:07, 54.26it/s]
Validation DataLoader 0:  52% 400/768 [00:07<00:06, 53.83it/s]
Validation DataLoader 0:  55% 420/768 [00:07<00:06, 54.23it/s]
Validation DataLoader 0:  57% 440/768 [00:08<00:06, 54.65it/s]
Validation DataLoader 0:  60% 460/768 [00:08<00:05, 54.89it/s]
Validation DataLoader 0:  62% 480/768 [00:08<00:05, 55.27it/s]
Validation DataLoader 0:  65% 500/768 [00:09<00:04, 55.52it/s]
Validation DataLoader 0:  68% 520/768 [00:09<00:04, 55.75it/s]
Validation DataLoader 0:  70% 540/768 [00:09<00:04, 56.04it/s]
Validation DataLoader 0:  73% 560/768 [00:09<00:03, 56.29it/s]
Validation DataLoader 0:  76% 580/768 [00:10<00:03, 56.60it/s]
Validation DataLoader 0:  78% 600/768 [00:10<00:02, 56.78it/s]
Validation DataLoader 0:  81% 620/768 [00:10<00:02, 56.98it/s]
Validation DataLoader 0:  83% 640/768 [00:11<00:02, 57.18it/s]
Validation DataLoader 0:  86% 660/768 [00:11<00:01, 57.25it/s]
Validation DataLoader 0:  89% 680/768 [00:11<00:01, 57.37it/s]
Validation DataLoader 0:  91% 700/768 [00:12<00:01, 57.48it/s]
Validation DataLoader 0:  94% 720/768 [00:12<00:00, 57.54it/s]
Validation DataLoader 0:  96% 740/768 [00:12<00:00, 57.65it/s]
Validation DataLoader 0:  99% 760/768 [00:13<00:00, 57.70it/s]
Validation DataLoader 0: 100% 768/768 [00:13<00:00, 57.86it/s][2026-03-02 23:27:40,913][spectra.trainer][INFO] - [Val] outcome: AUC=0.7745, PRC=0.1024, R=0.6459
[2026-03-02 23:27:40,920][spectra.trainer][INFO] - [Val] phase: ACC=0.3333, AUC=0.7739

Epoch 2: 100% 7016/7016 [05:02<00:00, 23.16it/s, GN=0.952, C=0.000, L=1.100, vL=0.528, outcome_AUC=0.774, outcome_PRC=0.102, outcome_R=0.646, AUC=0.774, phase_ACC=0.333, phase_AUC=0.774]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/768 [00:00<?, ?it/s]
Validation DataLoader 0:   3% 20/768 [00:00<00:12, 60.21it/s]
Validation DataLoader 0:   5% 40/768 [00:00<00:11, 63.60it/s]
Validation DataLoader 0:   8% 60/768 [00:00<00:11, 62.44it/s]
Validation DataLoader 0:  10% 80/768 [00:01<00:11, 61.77it/s]
Validation DataLoader 0:  13% 100/768 [00:01<00:10, 62.88it/s]
Validation DataLoader 0:  16% 120/768 [00:01<00:10, 62.53it/s]
Validation DataLoader 0:  18% 140/768 [00:02<00:10, 62.78it/s]
Validation DataLoader 0:  21% 160/768 [00:02<00:09, 63.33it/s]
Validation DataLoader 0:  23% 180/768 [00:02<00:09, 63.01it/s]
Validation DataLoader 0:  26% 200/768 [00:03<00:08, 63.14it/s]
Validation DataLoader 0:  29% 220/768 [00:03<00:08, 61.73it/s]
Validation DataLoader 0:  31% 240/768 [00:03<00:08, 60.17it/s]
Validation DataLoader 0:  34% 260/768 [00:04<00:08, 58.81it/s]
Validation DataLoader 0:  36% 280/768 [00:04<00:08, 57.52it/s]
Validation DataLoader 0:  39% 300/768 [00:05<00:08, 56.47it/s]
Validation DataLoader 0:  42% 320/768 [00:05<00:08, 55.52it/s]
Validation DataLoader 0:  44% 340/768 [00:06<00:07, 54.91it/s]
Validation DataLoader 0:  47% 360/768 [00:06<00:07, 54.12it/s]
Validation DataLoader 0:  49% 380/768 [00:07<00:07, 53.41it/s]
Validation DataLoader 0:  52% 400/768 [00:07<00:06, 53.48it/s]
Validation DataLoader 0:  55% 420/768 [00:07<00:06, 53.62it/s]
Validation DataLoader 0:  57% 440/768 [00:08<00:06, 53.92it/s]
Validation DataLoader 0:  60% 460/768 [00:08<00:05, 54.18it/s]
Validation DataLoader 0:  62% 480/768 [00:08<00:05, 54.58it/s]
Validation DataLoader 0:  65% 500/768 [00:09<00:04, 54.96it/s]
Validation DataLoader 0:  68% 520/768 [00:09<00:04, 55.21it/s]
Validation DataLoader 0:  70% 540/768 [00:09<00:04, 55.49it/s]
Validation DataLoader 0:  73% 560/768 [00:10<00:03, 55.71it/s]
Validation DataLoader 0:  76% 580/768 [00:10<00:03, 55.55it/s]
Validation DataLoader 0:  78% 600/768 [00:10<00:03, 55.75it/s]
Validation DataLoader 0:  81% 620/768 [00:11<00:02, 55.91it/s]
Validation DataLoader 0:  83% 640/768 [00:11<00:02, 55.95it/s]
Validation DataLoader 0:  86% 660/768 [00:11<00:01, 56.18it/s]
Validation DataLoader 0:  89% 680/768 [00:12<00:01, 56.44it/s]
Validation DataLoader 0:  91% 700/768 [00:12<00:01, 56.68it/s]
Validation DataLoader 0:  94% 720/768 [00:12<00:00, 56.84it/s]
Validation DataLoader 0:  96% 740/768 [00:12<00:00, 56.97it/s]
Validation DataLoader 0:  99% 760/768 [00:13<00:00, 57.15it/s]
Validation DataLoader 0: 100% 768/768 [00:13<00:00, 57.33it/s][2026-03-02 23:32:57,269][spectra.trainer][INFO] - [Val] outcome: AUC=0.7614, PRC=0.0934, R=0.5969
[2026-03-02 23:32:57,276][spectra.trainer][INFO] - [Val] phase: ACC=0.3348, AUC=0.7585

Epoch 3: 100% 7016/7016 [05:15<00:00, 22.27it/s, GN=0.986, C=0.000, L=1.070, vL=0.553, outcome_AUC=0.761, outcome_PRC=0.0934, outcome_R=0.597, AUC=0.760, phase_ACC=0.335, phase_AUC=0.758]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation   0% 0/? [00:00<?, ?it/s]         
Validation DataLoader 0:   0% 0/768 [00:00<?, ?it/s]
Validation DataLoader 0:   3% 20/768 [00:00<00:12, 58.73it/s]
Validation DataLoader 0:   5% 40/768 [00:00<00:12, 58.80it/s]
Validation DataLoader 0:   8% 60/768 [00:01<00:11, 59.89it/s]
Validation DataLoader 0:  10% 80/768 [00:01<00:11, 59.57it/s]
Validation DataLoader 0:  13% 100/768 [00:01<00:11, 60.10it/s]
Validation DataLoader 0:  16% 120/768 [00:01<00:10, 60.34it/s]
Validation DataLoader 0:  18% 140/768 [00:02<00:10, 60.14it/s]
Validation DataLoader 0:  21% 160/768 [00:02<00:10, 60.59it/s]
Validation DataLoader 0:  23% 180/768 [00:02<00:09, 60.32it/s]
Validation DataLoader 0:  26% 200/768 [00:03<00:09, 60.59it/s]
Validation DataLoader 0:  29% 220/768 [00:03<00:09, 60.67it/s]
Validation DataLoader 0:  31% 240/768 [00:03<00:08, 60.50it/s]
Validation DataLoader 0:  34% 260/768 [00:04<00:08, 60.53it/s]
Validation DataLoader 0:  36% 280/768 [00:04<00:08, 60.48it/s]
Validation DataLoader 0:  39% 300/768 [00:04<00:07, 60.18it/s]
Validation DataLoader 0:  42% 320/768 [00:05<00:07, 60.22it/s]
Validation DataLoader 0:  44% 340/768 [00:05<00:07, 60.11it/s]
Validation DataLoader 0:  47% 360/768 [00:06<00:06, 59.88it/s]
Validation DataLoader 0:  49% 380/768 [00:06<00:06, 59.94it/s]
Validation DataLoader 0:  52% 400/768 [00:06<00:06, 60.03it/s]
Validation DataLoader 0:  55% 420/768 [00:06<00:05, 60.03it/s]
Validation DataLoader 0:  57% 440/768 [00:07<00:05, 59.38it/s]
Validation DataLoader 0:  60% 460/768 [00:07<00:05, 58.75it/s]
Validation DataLoader 0:  62% 480/768 [00:08<00:04, 57.89it/s]
Validation DataLoader 0:  65% 500/768 [00:08<00:04, 57.11it/s]
Validation DataLoader 0:  68% 520/768 [00:09<00:04, 56.49it/s]
Validation DataLoader 0:  70% 540/768 [00:09<00:04, 55.97it/s]
Validation DataLoader 0:  73% 560/768 [00:10<00:03, 55.33it/s]
Validation DataLoader 0:  76% 580/768 [00:10<00:03, 54.77it/s]
Validation DataLoader 0:  78% 600/768 [00:11<00:03, 54.27it/s]
Validation DataLoader 0:  81% 620/768 [00:11<00:02, 54.31it/s]
Validation DataLoader 0:  83% 640/768 [00:11<00:02, 54.57it/s]
Validation DataLoader 0:  86% 660/768 [00:12<00:01, 54.57it/s]
Validation DataLoader 0:  89% 680/768 [00:12<00:01, 54.77it/s]
Validation DataLoader 0:  91% 700/768 [00:12<00:01, 54.97it/s]
Validation DataLoader 0:  94% 720/768 [00:13<00:00, 55.19it/s]
Validation DataLoader 0:  96% 740/768 [00:13<00:00, 55.45it/s]
Validation DataLoader 0:  99% 760/768 [00:13<00:00, 55.59it/s]
Validation DataLoader 0: 100% 768/768 [00:13<00:00, 55.75it/s][2026-03-02 23:38:26,197][spectra.trainer][INFO] - [Val] outcome: AUC=0.7359, PRC=0.0772, R=0.5430
[2026-03-02 23:38:26,204][spectra.trainer][INFO] - [Val] phase: ACC=0.3480, AUC=0.7318

Epoch 3: 100% 7016/7016 [05:28<00:00, 21.33it/s, GN=0.986, C=0.000, L=0.984, vL=0.585, outcome_AUC=0.736, outcome_PRC=0.0772, outcome_R=0.543, AUC=0.734, phase_ACC=0.348, phase_AUC=0.732]
[2026-03-02 23:38:26,491][spectra.train][INFO] - [Mission-Control] Mission Accomplished. ✓
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /content/SPECTRA/outputs/2026-03-02/23-16-47/wandb/offline-run-20260302_231648-x2w1epxg
wandb: Find logs at: outputs/2026-03-02/23-16-47/wandb/offline-run-20260302_231648-x2w1epxg/logs