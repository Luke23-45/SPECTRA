--- Testing SPECTRAModule with PCGrad on REAL Clinical Sepsis Data ---
2026-03-02 23:00:02,662 | INFO | [TRAIN] Loading Index: datasets/sepsis_clinical_28/train_index.json
2026-03-02 23:00:03,008 | INFO | [TRAIN] Piloting Mode: Subsetting to 5.0% (1774/35499 episodes)
2026-03-02 23:00:03,009 | INFO | [TRAIN] Initialized. Windows: 22,953 | Episodes: 1,189
2026-03-02 23:00:03,040 | INFO | [VAL] Loading Index: datasets/sepsis_clinical_28/val_index.json
2026-03-02 23:00:03,071 | INFO | [VAL] Piloting Mode: Subsetting to 5.0% (195/3904 episodes)
2026-03-02 23:00:03,071 | INFO | [VAL] Initialized. Windows: 2,409 | Episodes: 125
2026-03-02 23:00:03,092 | INFO | [SPECTRA] Initialized: backbone=shared_trunk, method=pcgrad, ALB=False, tasks=2
2026-03-02 23:00:03,096 | INFO | [SPECTRA] Task metrics initialized: ['outcome', 'phase']
GPU available: True (cuda), used: False
TPU available: False, using: 0 TPU cores
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Starting training on 22953 samples...
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
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

[Epoch 0 Val] Total Loss: 2.0276 | Outcome Val Loss: 0.7304 | Outcome AUC: N/A
/usr/local/lib/python3.12/dist-packages/torchmetrics/utilities/prints.py:43: UserWarning: No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score
  warnings.warn(*args, **kwargs)
/usr/local/lib/python3.12/dist-packages/torchmetrics/utilities/prints.py:43: UserWarning: No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.
  warnings.warn(*args, **kwargs)
2026-03-02 23:00:03,456 | INFO | [Val] outcome: AUC=0.0000, PRC=-0.0000, R=0.0000
/usr/local/lib/python3.12/dist-packages/torchmetrics/utilities/prints.py:43: UserWarning: No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score
  warnings.warn(*args, **kwargs)
2026-03-02 23:00:03,459 | INFO | [Val] phase: ACC=0.0000, AUC=0.0000
Epoch 0: 100% 359/359 [02:33<00:00,  2.64it/s, loss=0.5764, gn=0.7071, conf=0.0]
[Epoch 0 Val] Total Loss: 0.4011 | Outcome Val Loss: 0.2284 | Outcome AUC: N/A
2026-03-02 23:02:39,958 | INFO | [Val] outcome: AUC=0.7873, PRC=0.3063, R=0.0000
2026-03-02 23:02:39,962 | INFO | [Val] phase: ACC=0.3333, AUC=0.7837
Epoch 0: 100% 359/359 [02:36<00:00,  2.29it/s, loss=0.5764, gn=0.7071, conf=0.0]
Epoch 1: 100% 359/359 [02:35<00:00,  2.84it/s, loss=0.3550, gn=0.8604, conf=0.0]
[Epoch 1 Val] Total Loss: 0.3856 | Outcome Val Loss: 0.2164 | Outcome AUC: 0.7873
2026-03-02 23:05:18,226 | INFO | [Val] outcome: AUC=0.7893, PRC=0.3070, R=0.0000
2026-03-02 23:05:18,229 | INFO | [Val] phase: ACC=0.3333, AUC=0.7824
Epoch 1: 100% 359/359 [02:38<00:00,  2.27it/s, loss=0.3550, gn=0.8604, conf=0.0]
Epoch 2: 100% 359/359 [02:35<00:00,  2.73it/s, loss=0.4787, gn=0.2354, conf=8.0]
[Epoch 2 Val] Total Loss: 0.3638 | Outcome Val Loss: 0.1931 | Outcome AUC: 0.7893
2026-03-02 23:07:56,964 | INFO | [Val] outcome: AUC=0.7993, PRC=0.2884, R=0.0000
2026-03-02 23:07:56,969 | INFO | [Val] phase: ACC=0.3333, AUC=0.7875
Epoch 2: 100% 359/359 [02:38<00:00,  2.26it/s, loss=0.4787, gn=0.2354, conf=8.0]
Epoch 3: 100% 359/359 [02:37<00:00,  2.13it/s, loss=0.6344, gn=0.3572, conf=0.0]
[Epoch 3 Val] Total Loss: 0.3533 | Outcome Val Loss: 0.1815 | Outcome AUC: 0.7993
2026-03-02 23:10:37,190 | INFO | [Val] outcome: AUC=0.8105, PRC=0.2836, R=0.0000
2026-03-02 23:10:37,194 | INFO | [Val] phase: ACC=0.3333, AUC=0.7851
Epoch 3: 100% 359/359 [02:40<00:00,  2.24it/s, loss=0.6344, gn=0.3572, conf=0.0]
Epoch 4: 100% 359/359 [02:36<00:00,  2.48it/s, loss=0.4804, gn=0.9580, conf=0.0]
[Epoch 4 Val] Total Loss: 0.3553 | Outcome Val Loss: 0.1854 | Outcome AUC: 0.8105
2026-03-02 23:13:16,043 | INFO | [Val] outcome: AUC=0.8139, PRC=0.2784, R=0.0000
2026-03-02 23:13:16,046 | INFO | [Val] phase: ACC=0.3333, AUC=0.7877
Epoch 4: 100% 359/359 [02:38<00:00,  2.26it/s, loss=0.4804, gn=0.9580, conf=0.0]
`Trainer.fit` stopped: `max_epochs=5` reached.
Training completed successfully!