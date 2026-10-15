"""Tabular baselines on the MIMIC-IV landmark table, using the same 10 folds and metrics as the graph runner.

Usage: python tabular_mimic.py <features_L24.parquet> <results dir> <run prefix>
Requires splits.py and tabular_baselines.py from repro/ on the path. Writes per-fold JSON (aggregate metrics only).
"""
import json, sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from splits import paper_splits
from tabular_baselines import metrics

OUTCOME = {'home': 0, 'rehab': 1, 'death': 2}
EVENTS = ['nimodipine', 'paracetamol', 'morphine', 'norepinephrine', 'vasopressor', 'milrinone', 'evd', 'intubation',
          'coiling', 'clipping', 'angioplasty']


def main(features_path, results_dir, prefix):
    f = pd.read_parquet(features_path).reset_index(drop=True)          # same row order as the graphs' patients
    y = f['outcome'].map(OUTCOME).values
    X = f.drop(columns=['hadm_id', 'outcome']).assign(gender=lambda d: (d.gender == 'F').astype(int))
    event_cols = [c for c in X.columns if c in EVENTS or (c.endswith('_h') and c[:-2] in EVENTS)]
    models = {
        'LR': (lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000)), list(X.columns)),
        'GBM': (lambda: HistGradientBoostingClassifier(random_state=0), list(X.columns)),
        'AGE_GBM': (lambda: HistGradientBoostingClassifier(random_state=0), ['age']),
        'EVENTS_GBM': (lambda: HistGradientBoostingClassifier(random_state=0), event_cols),
    }
    for name, (make, cols) in models.items():
        out = Path(results_dir) / f'{prefix}_{name}'
        out.mkdir(parents=True, exist_ok=True)
        for k, (tr, va, te) in enumerate(paper_splits(y)):
            m = make().fit(X.iloc[tr][cols], y[tr])
            rec = dict(run=f'{prefix}_{name}', fold=k, **metrics(y[te], m.predict_proba(X.iloc[te][cols])))
            (out / f'fold{k}.json').write_text(json.dumps(rec, default=float))
        print(name, 'done', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
