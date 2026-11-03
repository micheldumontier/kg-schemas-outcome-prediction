| Run | auc_weighted | auc_macro | f1_macro | brier | log_loss |
|---|---|---|---|---|---|
| MIMIC24_caresm_ts_foldstd | 0.808 (0.032) | 0.814 (0.031) | 0.618 (0.040) | 0.468 (0.037) | 0.771 (0.068) |
| MIMIC24_sphn_ts_foldstd | 0.797 (0.026) | 0.804 (0.024) | 0.595 (0.034) | 0.490 (0.038) | 0.811 (0.082) |
| MIMIC24_sphn_real_foldstd | 0.813 (0.029) | 0.819 (0.027) | 0.632 (0.028) | 0.465 (0.034) | 0.762 (0.060) |
| MIMIC24_caresm_ts_global_cal | 0.809 (0.031) | 0.816 (0.029) | 0.625 (0.045) | 0.465 (0.035) | 0.765 (0.063) |
| MIMIC24_GBM_cal | 0.813 (0.023) | 0.822 (0.020) | 0.636 (0.024) | 0.554 (0.045) | 1.136 (0.121) |
| MIMIC24_LR_cal | 0.793 (0.023) | 0.801 (0.021) | 0.616 (0.031) | 0.492 (0.029) | 0.827 (0.077) |
| MIMIC24_EVENTS_GBM_cal | 0.682 (0.039) | 0.682 (0.037) | 0.452 (0.053) | 0.656 (0.061) | 1.219 (0.138) |
| MIMIC24_AGE_GBM_cal | 0.605 (0.029) | 0.604 (0.036) | 0.401 (0.037) | 0.637 (0.022) | 1.093 (0.077) |

Largest per-fold absolute difference in weighted AUC, calibrated rerun against paper run: 0.0083.

| Comparison | Metric | Difference | p (corrected) | 95% CI (corrected) |
|---|---|---|---|---|
| caresm_ts: fold-wise - global (paper run) | auc_weighted | -0.0000 | 0.991 | -0.0039 to +0.0039 |
| caresm_ts: fold-wise - global (paper run) | f1_macro | -0.0099 | 0.300 | -0.0303 to +0.0105 |
| sphn_ts: fold-wise - global (paper run) | auc_weighted | +0.0063 | 0.570 | -0.0180 to +0.0307 |
| sphn_ts: fold-wise - global (paper run) | f1_macro | +0.0015 | 0.949 | -0.0492 to +0.0521 |
| sphn_real: fold-wise - global (paper run) | auc_weighted | -0.0008 | 0.841 | -0.0100 to +0.0083 |
| sphn_real: fold-wise - global (paper run) | f1_macro | +0.0067 | 0.547 | -0.0175 to +0.0308 |
| caresm_ts: fold-wise - global (calibrated rerun) | auc_weighted | -0.0011 | 0.597 | -0.0056 to +0.0034 |
| caresm_ts: fold-wise - global (calibrated rerun) | auc_macro | -0.0013 | 0.564 | -0.0062 to +0.0036 |
| caresm_ts: fold-wise - global (calibrated rerun) | f1_macro | -0.0064 | 0.527 | -0.0286 to +0.0157 |
| caresm_ts: fold-wise - global (calibrated rerun) | brier | +0.0023 | 0.299 | -0.0025 to +0.0071 |
| caresm_ts: fold-wise - global (calibrated rerun) | log_loss | +0.0058 | 0.245 | -0.0047 to +0.0162 |
| caresm_ts: reproducibility, rerun - paper run | auc_weighted | +0.0011 | 0.475 | -0.0022 to +0.0044 |
| caresm_ts: reproducibility, rerun - paper run | f1_macro | -0.0035 | 0.590 | -0.0175 to +0.0106 |
| GBM - caresm_ts fold-wise | auc_weighted | +0.0048 | 0.682 | -0.0207 to +0.0303 |
| GBM - caresm_ts fold-wise | brier | +0.0865 | 0.002 | +0.0407 to +0.1323 |
| GBM - sphn_ts fold-wise | auc_weighted | +0.0153 | 0.114 | -0.0045 to +0.0350 |
| GBM - sphn_ts fold-wise | brier | +0.0644 | 0.022 | +0.0119 to +0.1169 |
| GBM - sphn_real fold-wise | auc_weighted | +0.0002 | 0.980 | -0.0186 to +0.0190 |
| GBM - sphn_real fold-wise | brier | +0.0889 | 0.002 | +0.0437 to +0.1340 |
