"""Does event timing add to event occurrence? Gradient boosting on the MIMIC-IV landmark tables, same 10 folds as the paper.

Usage: PYTHONPATH=repro python followup/kg/timing_mimic.py <dir with features_L{24,48,72}.parquet> <results dir>
Run from the repository root. splits.py and tabular_baselines.py come from repro/, tabular_mimic.py from followup/kg/.
For each landmark it fits four feature sets (event occurrence only, occurrence and first times, all features without
times, all features) and writes per-fold JSON (aggregate metrics only) plus summary.json with mean AUC, SD and paired
t-tests over folds. The corrected resampled t-tests of these comparisons are in timing_order.py.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from splits import paper_splits
from tabular_baselines import metrics
from tabular_mimic import EVENTS, OUTCOME

PAIRS = [('EV_OCC_TIME', 'EV_OCC'), ('ALL', 'ALL_NOTIME')]


def feature_sets(X):
    times = [f'{e}_h' for e in EVENTS]
    return {'EV_OCC': EVENTS, 'EV_OCC_TIME': EVENTS + times,
            'ALL_NOTIME': [c for c in X.columns if c not in times], 'ALL': list(X.columns)}


def run(features_path, results_dir, prefix):
    f = pd.read_parquet(features_path).reset_index(drop=True)
    y = f['outcome'].map(OUTCOME).values
    X = f.drop(columns=['hadm_id', 'outcome']).assign(gender=lambda d: (d.gender == 'F').astype(int))
    aucs = {}
    for name, cols in feature_sets(X).items():
        out = Path(results_dir) / f'{prefix}_{name}'
        out.mkdir(parents=True, exist_ok=True)
        aucs[name] = []
        for k, (tr, va, te) in enumerate(paper_splits(y)):
            m = HistGradientBoostingClassifier(random_state=0).fit(X.iloc[tr][cols], y[tr])
            rec = dict(run=f'{prefix}_{name}', fold=k, **metrics(y[te], m.predict_proba(X.iloc[te][cols])))
            rec.pop('pred_class_counts', None)
            (out / f'fold{k}.json').write_text(json.dumps(rec, default=float))
            aucs[name].append(rec['auc_weighted'])
    runs = {n: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for n, v in aucs.items()}
    pairs = [dict(comparison=f'{a} - {b}', diff=float(np.mean(np.subtract(aucs[a], aucs[b]))),
                  p=float(stats.ttest_rel(aucs[a], aucs[b]).pvalue)) for a, b in PAIRS]
    return dict(runs=runs, pairs=pairs)


if __name__ == '__main__':
    derived, results = sys.argv[1:3]
    summary = {f'L{L}': run(f'{derived}/features_L{L}.parquet', results, f'MIMIC{L}') for L in (24, 48, 72)}
    Path(results, 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
