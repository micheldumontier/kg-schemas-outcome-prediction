"""Pooled out-of-fold calibration: slope, calibration-in-the-large and ECE per class, with bootstrap CIs.

Usage: python calibration_slopes.py <probs dir> <output dir>
e.g. on the secure machine, from the repository root:
  python followup/kg/calibration_slopes.py ~/mimic-iv/derived/foldwise/probs followup/results/calibration
Reads <probs dir>/<run>/fold<k>.npz (idx, y, prob; written by repro/run_rgcn.py and followup/kg/tabular_mimic.py with
--save_probs, see foldwise_runs.sh), pools the 10 test folds so each patient is predicted once, and for every class
one-vs-rest computes
  slope  the coefficient of an unpenalised logistic regression of the class indicator on logit(p), p clipped to
         [1e-6, 1 - 1e-6] (ideal 1),
  citl   calibration-in-the-large, mean predicted probability minus observed frequency (ideal 0),
  ece    expected calibration error with 10 equal-frequency bins,
each with a 95% percentile bootstrap CI over patients (1,000 resamples, seed 0), plus pooled Brier score and log loss.
Writes calibration.md and calibration.json (aggregates only). Patient-level arrays never leave the secure machine.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

CLASSES = ['home', 'rehab', 'death']
RUNS = {'MIMIC24_caresm_ts_probs': 'R-GCN CARE-SM', 'MIMIC24_sphn_ts_probs': 'R-GCN SPHN',
        'MIMIC24_sphn_real_probs': 'R-GCN SPHN real', 'MIMIC24_GBM_probs': 'Gradient boosting',
        'MIMIC24_LR_probs': 'Logistic regression', 'MIMIC24_GBM_recal': 'Gradient boosting, Platt recalibrated'}
EPS, B, BINS = 1e-6, 1000, 10


def load_pooled(root, run, folds=10):
    idx, y, prob = [], [], []
    for k in range(folds):
        z = np.load(Path(root) / run / f'fold{k}.npz')
        idx.append(z['idx']); y.append(z['y']); prob.append(z['prob'])
    idx = np.concatenate(idx)
    assert len(np.unique(idx)) == len(idx), 'a patient appears in more than one test fold'
    order = np.argsort(idx)
    return np.concatenate(y)[order], np.concatenate(prob)[order]


def slope(ind, p):
    x = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(LogisticRegression(penalty=None, max_iter=1000).fit(x, ind).coef_[0, 0])


def citl(ind, p):
    return float(p.mean() - ind.mean())


def ece(ind, p, bins=BINS):
    order = np.argsort(p, kind='stable')
    parts = np.array_split(order, bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - ind[b].mean()) for b in parts if len(b)))


def pooled_loss(y, prob):
    onehot = np.eye(prob.shape[1])[y]
    q = np.clip(prob, EPS, 1)
    return float(((prob - onehot) ** 2).sum(1).mean()), float(-np.log(q[np.arange(len(y)), y] / q.sum(1)).mean())


def statistics(y, prob):
    out = {}
    for k, c in enumerate(CLASSES):
        ind = (y == k).astype(int)
        out[c] = dict(slope=slope(ind, prob[:, k]), citl=citl(ind, prob[:, k]), ece=ece(ind, prob[:, k]))
    out['brier'], out['log_loss'] = pooled_loss(y, prob)
    return out


def flat(s):
    return [s[c][m] for c in CLASSES for m in ('slope', 'citl', 'ece')] + [s['brier'], s['log_loss']]


def analyse(y, prob, boot=B, seed=0):
    rng = np.random.default_rng(seed)
    est = statistics(y, prob)
    reps = np.array([flat(statistics(y[i], prob[i])) for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    ci = dict(zip([(c, m) for c in CLASSES for m in ('slope', 'citl', 'ece')] + [('brier',), ('log_loss',)], zip(lo, hi)))
    res = {'n_patients': int(len(y))}
    for c in CLASSES:
        res[c] = {m: dict(est=est[c][m], ci=[float(x) for x in ci[(c, m)]]) for m in ('slope', 'citl', 'ece')}
    for m in ('brier', 'log_loss'):
        res[m] = dict(est=est[m], ci=[float(x) for x in ci[(m,)]])
    return res


def markdown(results):
    f = lambda d, s='': f"{d['est']:{s}.3f} ({d['ci'][0]:{s}.3f} to {d['ci'][1]:{s}.3f})"
    out = []
    for m, title, s in (('slope', 'Calibration slope (ideal 1)', ''), ('citl', 'Calibration-in-the-large, mean predicted minus observed (ideal 0)', '+'),
                        ('ece', 'Expected calibration error, 10 equal-frequency bins (ideal 0)', '')):
        out += [f'### {title}', '', '| Model | ' + ' | '.join(CLASSES) + ' |', '|---|' + '---|' * len(CLASSES)]
        out += [f'| {RUNS[r]} | ' + ' | '.join(f(v[c][m], s) for c in CLASSES) + ' |' for r, v in results.items()] + ['']
    out += ['### Pooled Brier score and log loss', '', '| Model | Brier | Log loss |', '|---|---|---|']
    out += [f"| {RUNS[r]} | {f(v['brier'])} | {f(v['log_loss'])} |" for r, v in results.items()]
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    results = {r: analyse(*load_pooled(root, r)) for r in RUNS}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'calibration.json').write_text(json.dumps(results, indent=1))
    (out_dir / 'calibration.md').write_text(markdown(results))
    print(markdown(results))
