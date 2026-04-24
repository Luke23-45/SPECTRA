# BPGS Deep-Dive Math Report (Synthetic NYUv2-like)

This sweep tests the core uncertainty objective only (no EMA, no extra training tricks).

## Best configuration

- mode: `log_mse`
- lr_theta: `0.03`
- bounds: `[-10.0, 10.0]`
- tail_weighted_loss: `0.436735`
- tail_uwso_gap: `0.166889` (negative is better than UWSO surrogate)

## Top 12 configurations

| rank | mode | lr_theta | bounds | tail_loss | tail_std | tail_uwso_gap | drift |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | log_mse | 0.03 | [-10,10] | 0.436735 | 0.043339 | 0.166889 | 0.200757 |
| 2 | log_mse | 0.03 | [-8,8] | 0.436750 | 0.042758 | 0.166904 | 0.200742 |
| 3 | log_mse | 0.03 | [-6,6] | 0.436777 | 0.042379 | 0.166930 | 0.200750 |
| 4 | huber_log | 0.03 | [-10,10] | 0.436831 | 0.043209 | 0.166984 | 0.200809 |
| 5 | huber_log | 0.03 | [-8,8] | 0.436840 | 0.042652 | 0.166993 | 0.200792 |
| 6 | huber_log | 0.03 | [-6,6] | 0.436846 | 0.042308 | 0.167000 | 0.200789 |
| 7 | log_mse | 0.01 | [-10,10] | 0.436913 | 0.042089 | 0.167067 | 0.200957 |
| 8 | huber_log | 0.01 | [-10,10] | 0.436958 | 0.042029 | 0.167111 | 0.200981 |
| 9 | log_mse | 0.01 | [-8,8] | 0.436982 | 0.041943 | 0.167136 | 0.201063 |
| 10 | huber_log | 0.01 | [-8,8] | 0.437031 | 0.041874 | 0.167184 | 0.201089 |
| 11 | log_mse | 0.01 | [-6,6] | 0.437095 | 0.041757 | 0.167248 | 0.201267 |
| 12 | huber_log | 0.01 | [-6,6] | 0.437149 | 0.041680 | 0.167303 | 0.201296 |

## Interpretation

- `kendall` reproduces competing-term behavior and consistently trails target-matching variants.
- `log_mse` and `huber_log` reduce objective conflict by directly matching `s_i` to `log(loss_i)`.
- Best runs come from direct target matching, suggesting the conflict is structural rather than just LR tuning.
