"""Summary of the aneurysm-repair sensitivity analysis (aggregates only).

Usage: python aggregate_cohort.py <cohort results dir> <mimic24 results dir> <output dir> <subset size>
  <home,rehab,death counts>
e.g. python followup/kg/aggregate_cohort.py <ANEU24 results> followup/results/mimic24 followup/results/cohort_aneurysm
  432 195,178,59   (the sizes printed by make_aneurysm_subset.py)
Copies the per-fold JSONs of the subset runs (ANEU24_*) to <output dir>/folds, and writes cohort_aneurysm.md and .json
with mean and SD over folds of weighted AUC and macro-F1, corrected paired tests (Nadeau-Bengio) and the full-cohort
values (MIMIC24_*, unpaired) beside them. Class counts below 11 are suppressed.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
from corrected_stats import corrected, K

MODELS = ['caresm_ts', 'sphn_ts', 'GBM', 'LR', 'AGE_GBM', 'EVENTS_GBM']
PAIRS = [('caresm_ts', 'sphn_ts'), ('GBM', 'caresm_ts'), ('GBM', 'sphn_ts'), ('caresm_ts', 'LR'), ('sphn_ts', 'LR')]


def metric(results_dir, run, key):
    files = sorted(Path(results_dir, run).glob('fold*.json'), key=lambda p: int(p.stem[4:]))
    if len(files) != K:
        raise ValueError(f'{run} has {len(files)} folds, expected {K}')
    return np.array([json.loads(f.read_text())[key] for f in files])


def suppress(n):
    return int(n) if n >= 11 else '<11'


def summarise(sub_dir, full_dir):
    out = dict(models={}, tests=[])
    for m in MODELS:
        row = {}
        for key in ('auc_weighted', 'f1_macro'):
            s, f = metric(sub_dir, f'ANEU24_{m}', key), metric(full_dir, f'MIMIC24_{m}', key)
            row[key] = dict(subset_mean=float(s.mean()), subset_sd=float(s.std(ddof=1)),
                            full_mean=float(f.mean()), full_sd=float(f.std(ddof=1)))
        out['models'][m] = row
    for a, b in PAIRS:
        for key in ('auc_weighted', 'f1_macro'):
            d = metric(sub_dir, f'ANEU24_{a}', key) - metric(sub_dir, f'ANEU24_{b}', key)
            out['tests'].append(dict(comparison=f'{a} - {b}', metric=key, **corrected(d)))
    return out


def markdown(res, n, counts):
    L = ['# Aneurysm-repair subset (24 h landmark)', '',
         'Selecting on repair during the admission uses information after the landmark: a sensitivity analysis of the '
         'cohort definition, not a deployable model.', '',
         f'Subset size {suppress(n)}; outcome classes ' + ', '.join(f'{k} {suppress(v)}' for k, v in counts.items()), '',
         '| Model | AUC subset | AUC full | macro-F1 subset | macro-F1 full |', '|---|---|---|---|---|']
    for m, r in res['models'].items():
        a, f = r['auc_weighted'], r['f1_macro']
        L.append(f"| {m} | {a['subset_mean']:.3f} ({a['subset_sd']:.3f}) | {a['full_mean']:.3f} ({a['full_sd']:.3f}) | "
                 f"{f['subset_mean']:.3f} ({f['subset_sd']:.3f}) | {f['full_mean']:.3f} ({f['full_sd']:.3f}) |")
    L += ['', 'Mean (SD) over 10 folds; full-cohort values are unpaired.', '',
          '| Comparison | Metric | Difference | Corrected p | 95% CI |', '|---|---|---|---|---|']
    for t in res['tests']:
        L.append(f"| {t['comparison']} | {t['metric']} | {t['diff']:+.3f} | {t['p_corr']:.3f} | "
                 f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] |")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    sub, full, out = map(Path, sys.argv[1:4])
    n = int(sys.argv[4])
    counts = dict(zip(('home', 'rehab', 'death'), map(int, sys.argv[5].split(','))))
    for run in sub.glob('ANEU24_*'):
        for f in run.glob('fold*.json'):
            d = json.loads(f.read_text())
            d.pop('pred_class_counts', None)
            (out / 'folds' / run.name).mkdir(parents=True, exist_ok=True)
            (out / 'folds' / run.name / f.name).write_text(json.dumps(d))
    res = summarise(sub, full)
    (out / 'cohort_aneurysm.json').write_text(json.dumps(res, indent=1))
    (out / 'cohort_aneurysm.md').write_text(markdown(res, n, counts))
    print((out / 'cohort_aneurysm.md').read_text())
