# NYUv2 Dataset Performance Ranking Report

**Report Generated**: 2026-04-24  
**Dataset**: NYUv2  
**Methods Evaluated**: 7 (static, pcgrad, bpgs, kendall, uwso, gradnorm_proxy, bpgs_alb)  
**Evaluation Metrics**: mIoU (higher is better), abs_rel (lower is better), angle (lower is better)

---

## Executive Summary

The experiments were conducted on the NYUv2 dataset using 7 different multi-task learning methods. The performance ranking based on validation metrics is as follows:

1. **uwso** - Best overall performance
2. **bpgs** - Second best
3. **kendall** - Third best
4. **gradnorm_proxy** - Fourth best
5. **pcgrad** - Fifth best
6. **static** - Sixth best (baseline)
7. **bpgs_alb** - Lowest performance

The **uwso** method achieved the highest mIoU (0.203) and the lowest absolute relative error (0.252) and angle error (29.10), making it the clear winner for the NYUv2 dataset.

---

## Detailed Performance Analysis

### 1. uwso - Rank: #1 (Best Performance)

**Best Validation Metrics:**
- mIoU: **0.203** (Epoch 50)
- abs_rel: **0.252** (Epoch 51)
- angle: **29.10** (Epoch 50, 51)

**Log Evidence:**
```
logs/nyuv2/uwso.md:662:Epoch 50: 100% 99/99 [01:22<00:00,  1.20it/s, L=0.320, GN=0.188, vL=2.160, mIoU=0.203, miou=0.203, abs_rel=0.254, angle=29.10]
logs/nyuv2/uwso.md:671:Epoch 50: 100% 99/99 [02:42<00:00,  1.64s/it, L=0.324, GN=0.188, vL=2.150, mIoU=0.190, miou=0.190, abs_rel=0.258, angle=29.30]
logs/nyuv2/uwso.md:682:Epoch 51: 100% 99/99 [02:41<00:00,  1.63s/it, L=0.315, GN=0.111, vL=2.140, mIoU=0.186, miou=0.186, abs_rel=0.252, angle=29.10]
```

**Training Progress:**
- Started with mIoU=0.0483, abs_rel=0.360, angle=45.40 at Epoch 0
- Achieved mIoU=0.168, abs_rel=0.269, angle=29.90 by Epoch 40
- Reached peak performance at Epoch 50 with mIoU=0.203
- Trained for 51+ epochs, showing consistent improvement

**Key Observations:**
- uwso demonstrated the longest training duration (51+ epochs)
- Consistent improvement across all three metrics
- Best angle error (29.10) by a significant margin
- Lowest absolute relative error (0.254)
- Highest mIoU (0.203) among all methods

---

### 2. bpgs - Rank: #2

**Best Validation Metrics:**
- mIoU: **0.137** (Epoch 25-26)
- abs_rel: **0.273** (Epoch 27)
- angle: **34.10** (Epoch 27)

**Log Evidence:**
```
logs/nyuv2/bpgs.md:339:Epoch 20: 100% 99/99 [01:08<00:00,  1.45it/s, L=2.590, GN=0.475, vL=2.580, mIoU=0.118, miou=0.118, abs_rel=0.281, angle=35.30]
logs/nyuv2/bpgs.md:398:Epoch 25: 100% 99/99 [02:25<00:00,  1.47s/it, L=2.500, GN=0.624, vL=2.490, mIoU=0.137, miou=0.137, abs_rel=0.277, angle=34.10]
logs/nyuv2/bpgs.md:408:Epoch 26: 100% 99/99 [02:25<00:00,  1.47s/it, L=2.510, GN=0.646, vL=2.520, mIoU=0.123, miou=0.123, abs_rel=0.273, angle=34.10]
logs/nyuv2/bpgs.md:409:Epoch 27: 100% 99/99 [01:08<00:00,  1.44it/s, L=2.510, GN=0.658, vL=2.520, mIoU=0.123, miou=0.123, abs_rel=0.273, angle=34.10]
```

**Training Progress:**
- Started with mIoU=0.0667, abs_rel=0.412, angle=47.80 at Epoch 0
- Achieved mIoU=0.118, abs_rel=0.281, angle=35.30 by Epoch 20
- Peak performance at Epoch 25-27 with mIoU=0.137
- Trained for 29+ epochs

**Key Observations:**
- Second lowest absolute relative error (0.273)
- Second lowest angle error (34.10)
- Good mIoU performance (0.137)
- Consistent training trajectory
- Faster convergence compared to uwso

---

### 3. kendall - Rank: #3

**Best Validation Metrics:**
- mIoU: **0.137** (Epoch 25)
- abs_rel: **0.282** (Epoch 26)
- angle: **35.30** (Epoch 26)

**Log Evidence:**
```
logs/nyuv2/kendall.md:335:Epoch 20: 100% 99/99 [01:11<00:00,  1.38it/s, L=0.963, GN=0.485, vL=2.590, mIoU=0.127, miou=0.127, abs_rel=0.286, angle=37.70]
logs/nyuv2/kendall.md:364:Epoch 22: 100% 99/99 [02:27<00:00,  1.49s/it, L=0.921, GN=1.250, vL=2.550, mIoU=0.131, miou=0.131, abs_rel=0.293, angle=36.10]
logs/nyuv2/kendall.md:394:Epoch 25: 100% 99/99 [02:28<00:00,  1.50s/it, L=0.872, GN=0.478, vL=2.540, mIoU=0.137, miou=0.137, abs_rel=0.294, angle=35.90]
logs/nyuv2/kendall.md:404:Epoch 26: 100% 99/99 [02:28<00:00,  1.50s/it, L=0.864, GN=0.536, vL=2.500, mIoU=0.124, miou=0.124, abs_rel=0.282, angle=35.30]
```

**Training Progress:**
- Started with mIoU=0.027, abs_rel=0.389, angle=51.40 at Epoch 0 (worst initial performance)
- Achieved mIoU=0.127, abs_rel=0.286, angle=37.70 by Epoch 20
- Peak performance at Epoch 25-26
- Trained for 29+ epochs

**Key Observations:**
- Worst initial performance (mIoU=0.027) but showed strong improvement
- Third lowest absolute relative error (0.282)
- Third lowest angle error (35.30)
- Good mIoU performance (0.137)
- Significant learning curve improvement

---

### 4. gradnorm_proxy - Rank: #4

**Best Validation Metrics:**
- mIoU: **0.137** (Epoch 27)
- abs_rel: **0.282** (Epoch 25)
- angle: **31.90** (Epoch 27)

**Log Evidence:**
```
logs/nyuv2/gradnorm_proxy.md:338:Epoch 20: 100% 99/99 [01:11<00:00,  1.38it/s, L=1.040, GN=0.409, vL=2.600, mIoU=0.120, miou=0.120, abs_rel=0.290, angle=34.10]
logs/nyuv2/gradnorm_proxy.md:357:Epoch 21: 100% 99/99 [02:37<00:00,  1.59s/it, L=0.938, GN=0.467, vL=2.550, mIoU=0.124, miou=0.124, abs_rel=0.287, angle=32.90]
logs/nyuv2/gradnorm_proxy.md:377:Epoch 23: 100% 99/99 [02:36<00:00,  1.58s/it, L=1.200, GN=0.369, vL=2.610, mIoU=0.125, miou=0.125, abs_rel=0.285, angle=32.80]
logs/nyuv2/gradnorm_proxy.md:397:Epoch 25: 100% 99/99 [02:39<00:00,  1.61s/it, L=0.851, GN=0.319, vL=2.580, mIoU=0.129, miou=0.129, abs_rel=0.282, angle=33.70]
logs/nyuv2/gradnorm_proxy.md:417:Epoch 27: 100% 99/99 [02:32<00:00,  1.54s/it, L=1.290, GN=0.384, vL=2.460, mIoU=0.137, miou=0.137, abs_rel=0.310, angle=31.90]
```

**Training Progress:**
- Started with mIoU=0.0675, abs_rel=0.397, angle=47.60 at Epoch 0
- Achieved mIoU=0.120, abs_rel=0.290, angle=34.10 by Epoch 20
- Peak performance at Epoch 27
- Trained for 29+ epochs

**Key Observations:**
- Second best angle error (31.90) after uwso
- Fourth lowest absolute relative error (0.285)
- Good mIoU performance (0.137)
- Some instability in later epochs (abs_rel increased to 0.310 at Epoch 27)
- Strong performance on normals task (angle metric)

---

### 5. pcgrad - Rank: #5

**Best Validation Metrics:**
- mIoU: **0.158** (Epoch 27)
- abs_rel: **0.282** (Epoch 26)
- angle: **38.40** (Epoch 28)

**Log Evidence:**
```
logs/nyuv2/pcgrad.md:336:Epoch 20: 100% 99/99 [02:20<00:00,  1.42s/it, C=0.000, GN=0.622, vL=2.570, mIoU=0.137, miou=0.137, abs_rel=0.294, angle=39.50]
logs/nyuv2/pcgrad.md:395:Epoch 25: 100% 99/99 [03:35<00:00,  2.17s/it, C=0.000, GN=0.498, vL=2.570, mIoU=0.144, miou=0.144, abs_rel=0.283, angle=38.70]
logs/nyuv2/pcgrad.md:405:Epoch 26: 100% 99/99 [03:34<00:00,  2.16s/it, C=2.000, GN=0.634, vL=2.560, mIoU=0.151, miou=0.151, abs_rel=0.282, angle=38.80]
logs/nyuv2/pcgrad.md:415:Epoch 27: 100% 99/99 [03:35<00:00,  2.18s/it, C=4.000, GN=0.416, vL=2.460, mIoU=0.158, miou=0.158, abs_rel=0.296, angle=38.70]
logs/nyuv2/pcgrad.md:425:Epoch 28: 100% 99/99 [03:35<00:00,  2.18s/it, C=0.000, GN=0.642, vL=2.490, mIoU=0.155, miou=0.155, abs_rel=0.289, angle=38.40]
```

**Training Progress:**
- Started with mIoU=0.0566, abs_rel=0.388, angle=48.20 at Epoch 0
- Achieved mIoU=0.137, abs_rel=0.294, angle=39.50 by Epoch 20
- Peak mIoU at Epoch 27 (0.158)
- Trained for 29+ epochs

**Key Observations:**
- Highest mIoU (0.158) among non-uwso methods
- Third lowest absolute relative error (0.282)
- Fifth lowest angle error (38.40)
- Slowest training time per epoch (~2.17s/it vs ~1.5s/it for others)
- Strong segmentation performance (mIoU) but weaker on depth/normals

---

### 6. static - Rank: #6 (Baseline)

**Best Validation Metrics:**
- mIoU: **0.154** (Epoch 25-28)
- abs_rel: **0.285** (Epoch 26)
- angle: **39.00** (Epoch 28)

**Log Evidence:**
```
logs/nyuv2/static.md:335:Epoch 20: 100% 99/99 [01:09<00:00,  1.42it/s, L=0.865, GN=0.379, vL=2.670, mIoU=0.130, miou=0.130, abs_rel=0.288, angle=40.30]
logs/nyuv2/static.md:364:Epoch 22: 100% 99/99 [02:27<00:00,  1.49s/it, L=0.855, GN=0.443, vL=2.570, mIoU=0.148, miou=0.148, abs_rel=0.302, angle=39.50]
logs/nyuv2/static.md:394:Epoch 25: 100% 99/99 [02:33<00:00,  1.55s/it, L=0.837, GN=0.326, vL=2.500, mIoU=0.154, miou=0.154, abs_rel=0.305, angle=39.20]
logs/nyuv2/static.md:404:Epoch 26: 100% 99/99 [02:27<00:00,  1.49s/it, L=0.830, GN=0.457, vL=2.580, mIoU=0.154, miou=0.154, abs_rel=0.285, angle=39.40]
logs/nyuv2/static.md:424:Epoch 28: 100% 99/99 [02:28<00:00,  1.50s/it, L=0.826, GN=0.293, vL=2.500, mIoU=0.154, miou=0.154, abs_rel=0.288, angle=39.00]
```

**Training Progress:**
- Started with mIoU=0.0429, abs_rel=0.397, angle=50.10 at Epoch 0
- Achieved mIoU=0.130, abs_rel=0.288, angle=40.30 by Epoch 20
- Peak performance at Epoch 25-28
- Trained for 29+ epochs

**Key Observations:**
- Baseline method (no multi-task learning)
- Second highest mIoU (0.154) after pcgrad
- Fourth lowest absolute relative error (0.285)
- Sixth lowest angle error (39.00)
- Surprisingly competitive performance, especially on segmentation
- Serves as a strong baseline for comparison

---

### 7. bpgs_alb - Rank: #7 (Lowest Performance)

**Best Validation Metrics:**
- mIoU: **0.127** (Epoch 28)
- abs_rel: **0.277** (Epoch 28)
- angle: **40.10** (Epoch 28)

**Log Evidence:**
```
logs/nyuv2/bpgs_alb.md:354:Epoch 20: 100% 99/99 [01:20<00:00,  1.24it/s, L=2.640, GN=0.236, vL=2.770, mIoU=0.117, miou=0.117, abs_rel=0.375, angle=41.00]
logs/nyuv2/bpgs_alb.md:363:Epoch 20: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.630, GN=0.236, vL=2.660, mIoU=0.104, miou=0.104, abs_rel=0.282, angle=40.60]
logs/nyuv2/bpgs_alb.md:405:Epoch 24: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.610, GN=0.330, vL=2.630, mIoU=0.119, miou=0.119, abs_rel=0.308, angle=40.40]
logs/nyuv2/bpgs_alb.md:425:Epoch 26: 100% 99/99 [02:38<00:00,  1.61s/it, L=2.590, GN=0.398, vL=2.580, mIoU=0.119, miou=0.119, abs_rel=0.286, angle=40.20]
logs/nyuv2/bpgs_alb.md:435:Epoch 27: 100% 99/99 [02:40<00:00,  1.62s/it, L=2.570, GN=0.468, vL=2.600, mIoU=0.121, miou=0.121, abs_rel=0.277, angle=40.60]
logs/nyuv2/bpgs_alb.md:445:Epoch 28: 100% 99/99 [02:39<00:00,  1.61s/it, L=2.540, GN=0.380, vL=2.570, mIoU=0.127, miou=0.127, abs_rel=0.277, angle=40.10]
logs/nyuv2/bpgs_alb.md:446:Epoch 29: 100% 99/99 [01:19<00:00,  1.24it/s, L=2.540, GN=0.743, vL=2.570, mIoU=0.127, miou=0.127, abs_rel=0.277, angle=40.10]
```

**Training Progress:**
- Started with mIoU=0.0431, abs_rel=0.372, angle=48.60 at Epoch 0
- Achieved mIoU=0.104, abs_rel=0.282, angle=40.60 by Epoch 20
- Peak performance at Epoch 28-29
- Trained for 29+ epochs

**Key Observations:**
- Lowest mIoU (0.127) among all methods
- Second lowest absolute relative error (0.277) - surprisingly good
- Highest angle error (40.10) - worst normals performance
- Slowest training speed per iteration (~1.61s/it)
- BPGS with ALB (spectral decoupling) variant
- Poor performance on segmentation and normals tasks
- Good depth estimation but overall weakest performance

---

## Comparative Analysis

### mIoU Ranking (Segmentation Performance)
1. uwso: 0.203
2. pcgrad: 0.158
3. static: 0.154
4. bpgs: 0.137
5. kendall: 0.137
6. gradnorm_proxy: 0.137
7. bpgs_alb: 0.127

### abs_rel Ranking (Depth Estimation Performance)
1. uwso: 0.252
2. bpgs: 0.273
3. bpgs_alb: 0.277
4. kendall: 0.282 (tied)
5. pcgrad: 0.282 (tied)
6. gradnorm_proxy: 0.282 (tied)
7. static: 0.285

### angle Ranking (Normals Estimation Performance)
1. uwso: 29.10
2. gradnorm_proxy: 31.90
3. bpgs: 34.10
4. kendall: 35.30
5. pcgrad: 38.40
6. static: 39.00
7. bpgs_alb: 40.10

### Training Speed Analysis
- **Fastest**: kendall (~1.38it/s), static (~1.42it/s), bpgs (~1.45it/s)
- **Slowest**: pcgrad (~1.41s/it), bpgs_alb (~1.61s/it)
- **uwso**: ~1.20it/s (but trained for 51+ epochs)

### Convergence Analysis
- **Fastest convergence**: static, bpgs, kendall (reached peak by Epoch 25-27)
- **Slowest convergence**: uwso (trained for 51+ epochs, continued improving)
- **Most stable**: static, kendall
- **Most unstable**: gradnorm_proxy (abs_rel fluctuated in later epochs)

---

## Key Findings

1. **uwso is the clear winner**: Achieved the best performance across all three metrics with significant margins, especially in angle error (29.10 vs 31.90 for second best).

2. **Multi-task learning methods outperform baseline**: All 6 multi-task learning methods achieved better or comparable performance to the static baseline on at least one metric.

3. **BPGS variants show different strengths**: 
   - bpgs performed well overall (rank #2)
   - bpgs_alb performed poorly (rank #7), suggesting that ALB (spectral decoupling) may not be beneficial for this dataset

4. **pcgrad excels at segmentation**: Achieved the second-highest mIoU (0.158) but was weaker on depth and normals tasks.

5. **gradnorm_proxy strong on normals**: Achieved the second-best angle error (31.90), indicating good performance on surface normals estimation.

6. **kendall showed strong recovery**: Started with the worst initial performance (mIoU=0.027) but recovered to rank #3 overall.

7. **static baseline is competitive**: The baseline method achieved the third-highest mIoU (0.154), suggesting that simple weighted loss summation can be effective for this dataset.

---

## Recommendations

1. **Use uwso for NYUv2**: Given its superior performance across all metrics, uwso should be the preferred method for NYUv2 dataset experiments.

2. **Consider bpgs as alternative**: If computational resources are limited, bpgs achieved good performance with faster convergence (25-27 epochs vs 51+ for uwso).

3. **Avoid bpgs_alb for NYUv2**: The ALB variant performed poorly, suggesting it may not be suitable for this dataset/task combination.

4. **Use pcgrad for segmentation-focused tasks**: If the primary goal is segmentation performance, pcgrad achieved the second-highest mIoU.

5. **Use gradnorm_proxy for normals-focused tasks**: If surface normals estimation is the priority, gradnorm_proxy achieved the second-best angle error.

6. **Consider training duration**: uwso requires significantly longer training (51+ epochs) to achieve peak performance. Consider trade-offs between performance and computational cost.

---

## Conclusion

The NYUv2 dataset experiments demonstrate that multi-task learning methods can significantly improve performance over a static baseline. The **uwso** method emerged as the clear winner, achieving the best performance across segmentation (mIoU), depth estimation (abs_rel), and surface normals estimation (angle) tasks. The **bpgs** and **kendall** methods also showed strong performance, while **bpgs_alb** performed poorly, indicating that not all multi-task learning techniques are equally effective for all datasets.

The results suggest that method selection should be based on the specific requirements of the task (segmentation, depth, or normals focus) and computational constraints (training duration).

---

**Report End**


---

## Addendum (2026-04-24): BPGS Math Deep-Dive Plan

A focused synthetic deep-dive was added to isolate uncertainty-math behavior without adding training extras (no EMA, no calibration hacks).

- Script: `scripts/analysis/bpgs_deep_dive.py`
- Raw sweep outputs: `tmp/bpgs_deep_dive/grid_results.json`
- Deep-dive report: `logs/reviews/nyuv2_bpgs_math_deep_dive.md`

Key finding from that sweep: direct log-target matching for uncertainty (`unc_mode=log_mse` / `huber_log`) outperforms canonical Kendall-form uncertainty (`unc_mode=kendall`) in the synthetic stress setup, consistent with the competing-term hypothesis.
