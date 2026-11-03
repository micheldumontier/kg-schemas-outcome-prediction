"""Tabular baselines on the MIMIC-IV landmark table, using the same 10 folds and metrics as the graph runner.

Usage: python tabular_mimic.py <features_L24.parquet> <results dir> <run prefix> [--calibration] [--models LR,GBM,...]
                                [--suffix <run name suffix>] [--save_probs <dir>]
With --calibration the runs are named <prefix>_<model>_cal and each fold also holds Brier score, log loss and macro AUC.
GBM_recal (gradient boosting with Platt recalibration fitted on the validation split) runs only when named in --models.
--save_probs writes patient-level test probabilities for calibration_slopes.py; they never leave the secure machine.
Requires splits.py, tabular_baselines.py and calibration.py from repro/ on the path (PYTHONPATH=repro). Writes
per-fold JSON (aggregate metrics only).
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
from calibration import calibration_metrics, save_probs, platt_recalibrate

OUTCOME = {'home': 0, 'rehab': 1, 'death': 2}
EVENTS = ['nimodipine', 'paracetamol', 'morphine', 'norepinephrine', 'vasopressor', 'milrinone', 'evd', 'intubation',
          'coiling', 'clipping', 'angioplasty']


def main(features_path, results_dir, prefix, calibration=False, probs_dir=None, only=None, suffix=None):
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
    models['GBM_recal'] = (models['GBM'][0], models['GBM'][1])   # GBM plus Platt recalibration fitted on the validation split
    for name, (make, cols) in models.items():
        if (only is None and name == 'GBM_recal') or (only is not None and name not in only):
            continue
        run = f'{prefix}_{name}' + (suffix if suffix is not None else ('_cal' if calibration else ''))
        out = Path(results_dir) / run
        out.mkdir(parents=True, exist_ok=True)
        for k, (tr, va, te) in enumerate(paper_splits(y)):
            m = make().fit(X.iloc[tr][cols], y[tr])
            prob = m.predict_proba(X.iloc[te][cols])
            if name == 'GBM_recal':
                prob = platt_recalibrate(m.predict_proba(X.iloc[va][cols]), y[va], prob)
            rec = dict(run=run, fold=k, **metrics(y[te], prob))
            if calibration or probs_dir:
                rec.update(calibration_metrics(y[te], prob))
            if probs_dir:
                save_probs(probs_dir, run, k, te, y[te], prob)
            (out / f'fold{k}.json').write_text(json.dumps(rec, default=float))
        print(name, 'done', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('features'); ap.add_argument('results'); ap.add_argument('prefix')
    ap.add_argument('--calibration', action='store_true')
    ap.add_argument('--save_probs', default=None)
    ap.add_argument('--models', default=None, help='comma list of LR,GBM,AGE_GBM,EVENTS_GBM,GBM_recal')
    ap.add_argument('--suffix', default=None, help='run name suffix (default _cal with --calibration)')
    a = ap.parse_args()
    main(a.features, a.results, a.prefix, a.calibration, a.save_probs, a.models.split(',') if a.models else None, a.suffix)
