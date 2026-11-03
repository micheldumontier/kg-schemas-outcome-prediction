### Calibration slope (ideal 1)

| Model | home | rehab | death |
|---|---|---|---|
| R-GCN CARE-SM | 0.972 (0.874 to 1.093) | 1.041 (0.869 to 1.208) | 0.999 (0.881 to 1.157) |
| R-GCN SPHN | 0.840 (0.746 to 0.956) | 0.615 (0.497 to 0.739) | 0.828 (0.723 to 0.965) |
| R-GCN SPHN real | 0.893 (0.810 to 1.000) | 0.835 (0.708 to 0.966) | 0.879 (0.783 to 1.003) |
| Gradient boosting | 0.340 (0.306 to 0.377) | 0.226 (0.194 to 0.261) | 0.375 (0.337 to 0.422) |
| Logistic regression | 0.741 (0.648 to 0.839) | 0.525 (0.395 to 0.642) | 0.672 (0.559 to 0.795) |
| Gradient boosting, Platt recalibrated | 1.079 (0.974 to 1.197) | 0.784 (0.667 to 0.916) | 1.131 (1.018 to 1.268) |

### Calibration-in-the-large, mean predicted minus observed (ideal 0)

| Model | home | rehab | death |
|---|---|---|---|
| R-GCN CARE-SM | -0.017 (-0.035 to +0.001) | +0.003 (-0.021 to +0.027) | +0.014 (-0.006 to +0.033) |
| R-GCN SPHN | +0.003 (-0.017 to +0.023) | -0.003 (-0.028 to +0.021) | +0.001 (-0.018 to +0.019) |
| R-GCN SPHN real | +0.005 (-0.013 to +0.024) | -0.012 (-0.035 to +0.011) | +0.006 (-0.012 to +0.025) |
| Gradient boosting | +0.022 (+0.001 to +0.042) | +0.011 (-0.016 to +0.037) | -0.033 (-0.050 to -0.014) |
| Logistic regression | -0.001 (-0.020 to +0.018) | +0.000 (-0.024 to +0.024) | +0.001 (-0.018 to +0.019) |
| Gradient boosting, Platt recalibrated | -0.008 (-0.027 to +0.011) | +0.009 (-0.016 to +0.032) | -0.000 (-0.019 to +0.017) |

### Expected calibration error, 10 equal-frequency bins (ideal 0)

| Model | home | rehab | death |
|---|---|---|---|
| R-GCN CARE-SM | 0.027 (0.022 to 0.054) | 0.031 (0.024 to 0.059) | 0.035 (0.022 to 0.053) |
| R-GCN SPHN | 0.026 (0.021 to 0.052) | 0.055 (0.042 to 0.082) | 0.031 (0.022 to 0.053) |
| R-GCN SPHN real | 0.026 (0.020 to 0.051) | 0.035 (0.028 to 0.068) | 0.031 (0.022 to 0.056) |
| Gradient boosting | 0.132 (0.114 to 0.153) | 0.208 (0.186 to 0.233) | 0.093 (0.075 to 0.113) |
| Logistic regression | 0.042 (0.030 to 0.064) | 0.072 (0.058 to 0.100) | 0.031 (0.026 to 0.054) |
| Gradient boosting, Platt recalibrated | 0.027 (0.021 to 0.051) | 0.050 (0.036 to 0.075) | 0.018 (0.017 to 0.043) |

### Pooled Brier score and log loss

| Model | Brier | Log loss |
|---|---|---|
| R-GCN CARE-SM | 0.463 (0.445 to 0.484) | 0.761 (0.730 to 0.794) |
| R-GCN SPHN | 0.494 (0.471 to 0.518) | 0.813 (0.773 to 0.851) |
| R-GCN SPHN real | 0.461 (0.440 to 0.484) | 0.754 (0.719 to 0.791) |
| Gradient boosting | 0.554 (0.515 to 0.592) | 1.136 (1.038 to 1.229) |
| Logistic regression | 0.492 (0.469 to 0.519) | 0.824 (0.778 to 0.879) |
| Gradient boosting, Platt recalibrated | 0.463 (0.441 to 0.485) | 0.759 (0.728 to 0.791) |
