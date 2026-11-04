# Aneurysm-repair subset (24 h landmark)

Selecting on repair during the admission uses information after the landmark: a sensitivity analysis of the cohort definition, not a deployable model.

Subset size 432; outcome classes home 195, rehab 178, death 59

| Model | AUC subset | AUC full | macro-F1 subset | macro-F1 full |
|---|---|---|---|---|
| caresm_ts | 0.765 (0.083) | 0.808 (0.032) | 0.510 (0.116) | 0.628 (0.044) |
| sphn_ts | 0.718 (0.067) | 0.791 (0.032) | 0.470 (0.076) | 0.594 (0.046) |
| GBM | 0.801 (0.034) | 0.813 (0.023) | 0.557 (0.064) | 0.636 (0.024) |
| LR | 0.806 (0.057) | 0.793 (0.023) | 0.601 (0.074) | 0.616 (0.031) |
| AGE_GBM | 0.631 (0.055) | 0.605 (0.029) | 0.375 (0.052) | 0.401 (0.037) |
| EVENTS_GBM | 0.673 (0.051) | 0.682 (0.039) | 0.442 (0.063) | 0.452 (0.053) |

Mean (SD) over 10 folds; full-cohort values are unpaired.

| Comparison | Metric | Difference | Corrected p | 95% CI |
|---|---|---|---|---|
| caresm_ts - sphn_ts | auc_weighted | +0.048 | 0.129 | [-0.017, +0.112] |
| caresm_ts - sphn_ts | f1_macro | +0.041 | 0.315 | [-0.046, +0.127] |
| GBM - caresm_ts | auc_weighted | +0.035 | 0.305 | [-0.038, +0.108] |
| GBM - caresm_ts | f1_macro | +0.047 | 0.328 | [-0.056, +0.150] |
| GBM - sphn_ts | auc_weighted | +0.083 | 0.015 | [+0.020, +0.146] |
| GBM - sphn_ts | f1_macro | +0.088 | 0.001 | [+0.044, +0.131] |
| caresm_ts - LR | auc_weighted | -0.041 | 0.304 | [-0.125, +0.044] |
| caresm_ts - LR | f1_macro | -0.091 | 0.127 | [-0.213, +0.032] |
| sphn_ts - LR | auc_weighted | -0.088 | 0.035 | [-0.169, -0.008] |
| sphn_ts - LR | f1_macro | -0.131 | 0.011 | [-0.225, -0.038] |
