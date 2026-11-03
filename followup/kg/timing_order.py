"""Does explicit event order or interval information add to event occurrence? Gradient boosting on the MIMIC-IV
landmark tables, same 10 folds, model and metrics as the paper (exploratory follow-up to timing_mimic.py).

Usage: PYTHONPATH=repro python followup/kg/timing_order.py <dir with features_L{24,48,72}.parquet> <results dir>
Run from the repository root. splits.py and tabular_baselines.py come from repro/, tabular_mimic.py from followup/kg/.
Writes per-fold JSON and summary.json with aggregate metrics only; counts below 11 are suppressed. Paired comparisons
use the corrected resampled t-test, as in corrected_stats.py.
Feature sets (all derived from the first-time columns <event>_h, hours from admission, missing when the event did not
occur before the landmark):
  EV_OCC        occurrence of the 11 events (reference)
  EV_OCC_TIME   occurrence plus first times (as in timing_mimic.py)
  EV_ORDER      occurrence plus, for every pair of events, +1 / -1 / 0 when A is before / after / at the same time as B
  EV_INTERVAL   occurrence plus, for every pair, the signed hours from A to B
  EV_CLIN       occurrence plus clinically motivated timing: hours to aneurysm securing (coiling or clipping), securing
                before drain, securing before intubation, drain before intubation, hours to the first event, hours from
                the last event to the landmark, and the number of events
  ALL_NOTIME, ALL_NOTIME_CLIN, ALL_NOTIME_INTERVAL   the same additions on top of all non-time features
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from splits import paper_splits
from tabular_baselines import metrics
from tabular_mimic import EVENTS, OUTCOME

K, RATIO = 10, 0.1 / 0.8      # same correction as corrected_stats.py
PAIRS = [('EV_OCC_TIME', 'EV_OCC'), ('EV_ORDER', 'EV_OCC'), ('EV_INTERVAL', 'EV_OCC'), ('EV_CLIN', 'EV_OCC'),
         ('ALL_NOTIME_CLIN', 'ALL_NOTIME'), ('ALL_NOTIME_INTERVAL', 'ALL_NOTIME')]


def sup(n):
    return int(n) if n >= 11 else '<11'


def derived(f, landmark):
    t = f[[f'{e}_h' for e in EVENTS]].set_axis(EVENTS, axis=1)
    order, interval = {}, {}
    for a, b in itertools.combinations(EVENTS, 2):
        both = t[a].notna() & t[b].notna()
        order[f'ord_{a}_{b}'] = np.sign(t[b] - t[a]).where(both)
        interval[f'int_{a}_{b}'] = (t[b] - t[a]).where(both)
    secure = t[['coiling', 'clipping']].min(axis=1)
    clin = pd.DataFrame({
        'secure_h': secure,
        'secure_before_evd': np.sign(t['evd'] - secure).where(secure.notna() & t['evd'].notna()),
        'secure_before_intubation': np.sign(t['intubation'] - secure).where(secure.notna() & t['intubation'].notna()),
        'evd_before_intubation': np.sign(t['intubation'] - t['evd']).where(t['evd'].notna() & t['intubation'].notna()),
        'first_event_h': t.min(axis=1),
        'last_event_to_landmark_h': landmark - t.max(axis=1),
        'n_events': f[EVENTS].sum(axis=1)})
    counts = {'secure_and_evd': sup(clin['secure_before_evd'].notna().sum()),
              'secure_and_intubation': sup(clin['secure_before_intubation'].notna().sum()),
              'evd_and_intubation': sup(clin['evd_before_intubation'].notna().sum()),
              'any_event': sup(clin['first_event_h'].notna().sum()),
              'order_pairs_with_11_or_more_patients': int(sum(v.notna().sum() >= 11 for v in order.values())),
              'tied_pairs_share': float(np.nanmean([(v == 0).sum() / max(v.notna().sum(), 1) for v in order.values()
                                                    if v.notna().sum() >= 11]))}
    keep = [k for k, v in order.items() if v.notna().sum() >= 11]     # pairs seen in at least 11 patients
    order = {k: order[k] for k in keep}
    interval = {k.replace('ord_', 'int_', 1): interval[k.replace('ord_', 'int_', 1)] for k in keep}
    return pd.DataFrame(order), pd.DataFrame(interval), clin, counts


def corrected(d):
    m, s2 = float(np.mean(d)), float(np.var(d, ddof=1))
    se = np.sqrt((1 / K + RATIO) * s2)
    t = m / se if se > 0 else 0.0
    q = stats.t.ppf(0.975, K - 1)
    return dict(diff=m, p_corr=float(2 * stats.t.sf(abs(t), K - 1)), ci=[m - q * se, m + q * se])


def run(features_path, results_dir, landmark):
    f = pd.read_parquet(features_path).reset_index(drop=True)
    y = f['outcome'].map(OUTCOME).values
    X = f.drop(columns=['hadm_id', 'outcome']).assign(gender=lambda d: (d.gender == 'F').astype(int))
    order, interval, clin, counts = derived(X, landmark)
    times = [f'{e}_h' for e in EVENTS]
    notime = X[[c for c in X.columns if c not in times]]
    sets = {'EV_OCC': X[EVENTS], 'EV_OCC_TIME': X[EVENTS + times],
            'EV_ORDER': pd.concat([X[EVENTS], order], axis=1), 'EV_INTERVAL': pd.concat([X[EVENTS], interval], axis=1),
            'EV_CLIN': pd.concat([X[EVENTS], clin], axis=1), 'ALL_NOTIME': notime,
            'ALL_NOTIME_CLIN': pd.concat([notime, clin], axis=1),
            'ALL_NOTIME_INTERVAL': pd.concat([notime, interval], axis=1)}
    aucs = {}
    for name, Z in sets.items():
        out = Path(results_dir) / f'MIMIC{landmark}_{name}'
        out.mkdir(parents=True, exist_ok=True)
        aucs[name] = []
        for k, (tr, va, te) in enumerate(paper_splits(y)):
            m = HistGradientBoostingClassifier(random_state=0).fit(Z.iloc[tr], y[tr])
            rec = dict(run=f'MIMIC{landmark}_{name}', fold=k, **metrics(y[te], m.predict_proba(Z.iloc[te])))
            rec.pop('pred_class_counts', None)
            (out / f'fold{k}.json').write_text(json.dumps(rec, default=float))
            aucs[name].append(rec['auc_weighted'])
        print(landmark, name, round(float(np.mean(aucs[name])), 4), flush=True)
    runs = {n: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n_features=int(sets[n].shape[1]))
            for n, v in aucs.items()}
    pairs = [dict(comparison=f'{a} - {b}', **corrected(np.subtract(aucs[a], aucs[b]))) for a, b in PAIRS]
    return dict(n_patients=sup(len(f)), counts=counts, runs=runs, pairs=pairs)


if __name__ == '__main__':
    src, results = sys.argv[1:3]
    summary = {f'L{L}': run(f'{src}/features_L{L}.parquet', results, L) for L in (24, 48, 72)}
    Path(results, 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
