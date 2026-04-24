# NYUv2 BPGS Deep-Dive Report

Date: 2026-04-24

## Executive answer: Is scale-normalized BPGS "cheating"?
Short answer: **No, if and only if it is reported as a new method variant and compared under identical training budget/protocol.**

It **would** be unfair if we silently changed the default BPGS and reported it as original BPGS.

To avoid that, this repo now keeps:
- `method=bpgs` as the raw baseline (legacy math), and
- `method=bpgs_scaleinv` as the proposed variant.

## Scope
We ran an in-depth math/optimization sweep with simulated NYUv2-like multi-task dynamics to identify a fundamental BPGS weakness before spending full-data training budget.

Artifacts:
- Sweep script: `temp/bpgs_math_lab/bpgs_variant_sweep.py`
- Raw sweep table: `temp/bpgs_math_lab/results/variant_sweep.csv`
- Ranking summary: `temp/bpgs_math_lab/results/summary.json`

## Hypothesis tested
Raw uncertainty updates in BPGS are sensitive to absolute loss units and can skew task weighting when scales drift.

## Variants evaluated
12 combinations:
- Normalization: `raw`, `mean`, `max`, `geo`
- Linear coefficient: `0.35`, `0.5`, `0.65`

## Key result from simulation
Best mean score over 7 seeds:
- **`max_c0.65`** score **1.364944**

Worst mean score over 7 seeds:
- `raw_c0.65` score 1.370483

Relative improvement (best vs worst in stress simulation):
- `0.404%`

Top-3 variants are all `max` normalized.

## Why this is scientifically defensible
- Uncertainty weighting literature already highlights sensitivity to task scales/units.
- Our change does **not** add extra supervision, data, EMA, or extra optimizer passes.
- It modifies only the uncertainty-objective math and keeps the same split optimization structure.

## Fair-comparison protocol (must follow)
1. Run **both** `bpgs` and `bpgs_scaleinv` with identical:
   - architecture/backbone/heads
   - epochs, batch size, LR schedules, augmentations
   - seeds and evaluation code.
2. Report per-task metrics (mIoU, AbsRel, mean-angle) and aggregate deltas.
3. Label methods explicitly in tables/plots (`BPGS-raw` vs `BPGS-scaleinv`).
4. Do not mix best-seed cherry picks across methods; use the same seed set.

## Decision gate before claiming improvement
Promote `bpgs_scaleinv` only if it wins on average across agreed seeds without harming any critical task beyond tolerance.
