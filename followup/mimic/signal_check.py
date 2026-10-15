"""Preliminary signal check: 5-fold CV AUC per landmark. Prints aggregate metrics only."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sys
D = sys.argv[1] if len(sys.argv) > 1 else '/home/coder/mimic-iv/derived'
for L in (24, 48, 72):
    f = pd.read_parquet(f'{D}/features_L{L}.parquet')
    y = f['outcome'].map({'home': 0, 'rehab': 1, 'death': 2}).values
    X = f.drop(columns=['hadm_id', 'outcome']).assign(gender=lambda d: (d.gender == 'F').astype(int))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name, m, cols in [('LR all', make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000)), X.columns),
                          ('GBM all', HistGradientBoostingClassifier(random_state=0), X.columns),
                          ('GBM age only', HistGradientBoostingClassifier(random_state=0), ['age']),
                          ('GBM events only', HistGradientBoostingClassifier(random_state=0), [c for c in X.columns if c in
                              ('nimodipine','paracetamol','morphine','norepinephrine','vasopressor','milrinone','evd','intubation','coiling','clipping','angioplasty')
                              or c.endswith('_h')])]:
        p = cross_val_predict(m, X[list(cols)], y, cv=cv, method='predict_proba')
        res[name] = round(roc_auc_score(y, p, multi_class='ovr', average='weighted'), 3)
    print(f'landmark {L} h (n={len(f)}):', res)
