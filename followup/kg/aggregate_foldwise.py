"""Summarise the fold-wise standardisation and calibration runs (aggregates only).

Usage: python aggregate_foldwise.py <foldwise results dir> <paper mimic24 results dir> <output dir>
e.g. from the repository root: python followup/kg/aggregate_foldwise.py ~/mimic-iv/derived/foldwise/results
  ~/mimic-iv/derived/results followup/results/foldwise   (runs of foldwise_runs.sh and tabular_mimic.py --calibration)
Copies the per-fold JSONs of the foldwise runs to <output dir>/<run>/ without pred_class_counts, then writes
<output dir>/foldwise.md and foldwise.json: mean and SD over folds of weighted AUC, macro AUC, macro-F1, Brier and log
loss per run, and Nadeau-Bengio corrected paired tests (see corrected_stats.py) of
  - each fold-wise graph against the paper's global standardisation (weighted AUC and macro-F1; for caresm_ts also
    macro AUC, Brier and log loss against the calibrated rerun with global standardisation),
  - the calibrated rerun with global standardisation against the paper's run (reproducibility check),
  - gradient boosting against each fold-wise graph on weighted AUC and Brier.
"""
import json, sys
from pathlib import Path
import numpy as np
from corrected_stats import corrected, K
from sanitise_results import sanitise

METRICS = ['auc_weighted', 'auc_macro', 'f1_macro', 'brier', 'log_loss']
GRAPH_RUNS = ['caresm_ts', 'sphn_ts', 'sphn_real']


def load(results_dir, run):
    files = sorted(Path(results_dir, run).glob('fold*.json'), key=lambda p: int(p.stem[4:]))
    if len(files) != K:
        raise ValueError(f'{run} has {len(files)} folds, expected {K}')
    return [json.loads(f.read_text()) for f in files]


def column(recs, metric):
    return np.array([r[metric] for r in recs])


def summary(recs):
    out = {}
    for m in METRICS:
        if all(m in r for r in recs):
            v = column(recs, m)
            out[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
    return out


def paired(label, metric, a, b):
    return dict(comparison=label, metric=metric, **corrected(column(a, metric) - column(b, metric)))


def build(results_dir, paper_dir):
    runs = {f'MIMIC24_{g}_foldstd': load(results_dir, f'MIMIC24_{g}_foldstd') for g in GRAPH_RUNS}
    runs['MIMIC24_caresm_ts_global_cal'] = load(results_dir, 'MIMIC24_caresm_ts_global_cal')
    for t in ('GBM', 'LR', 'EVENTS_GBM', 'AGE_GBM'):
        runs[f'MIMIC24_{t}_cal'] = load(results_dir, f'MIMIC24_{t}_cal')
    paper = {g: load(paper_dir, f'MIMIC24_{g}') for g in GRAPH_RUNS}
    tests = []
    for g in GRAPH_RUNS:
        fs = runs[f'MIMIC24_{g}_foldstd']
        for m in ('auc_weighted', 'f1_macro'):
            tests.append(paired(f'{g}: fold-wise - global (paper run)', m, fs, paper[g]))
    gc, fs = runs['MIMIC24_caresm_ts_global_cal'], runs['MIMIC24_caresm_ts_foldstd']
    for m in METRICS:
        tests.append(paired('caresm_ts: fold-wise - global (calibrated rerun)', m, fs, gc))
    for m in ('auc_weighted', 'f1_macro'):
        tests.append(paired('caresm_ts: reproducibility, rerun - paper run', m, gc, paper['caresm_ts']))
    rerun_diff = float(np.abs(column(gc, 'auc_weighted') - column(paper['caresm_ts'], 'auc_weighted')).max())
    gbm = runs['MIMIC24_GBM_cal']
    for g in GRAPH_RUNS:
        for m in ('auc_weighted', 'brier'):
            tests.append(paired(f'GBM - {g} fold-wise', m, gbm, runs[f'MIMIC24_{g}_foldstd']))
    return runs, tests, rerun_diff


def markdown(runs, tests, rerun_diff):
    out = ['| Run | ' + ' | '.join(METRICS) + ' |', '|---|' + '---|' * len(METRICS)]
    for name, recs in runs.items():
        s = summary(recs)
        out.append(f'| {name} | ' + ' | '.join(f"{s[m]['mean']:.3f} ({s[m]['sd']:.3f})" if m in s else '' for m in METRICS) + ' |')
    out += ['', f'Largest per-fold absolute difference in weighted AUC, calibrated rerun against paper run: {rerun_diff:.4f}.', '',
            '| Comparison | Metric | Difference | p (corrected) | 95% CI (corrected) |', '|---|---|---|---|---|']
    for t in tests:
        out.append(f"| {t['comparison']} | {t['metric']} | {t['diff']:+.4f} | {t['p_corr']:.3f} | "
                   f"{t['ci'][0]:+.4f} to {t['ci'][1]:+.4f} |")
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    results_dir, paper_dir, out_dir = map(Path, sys.argv[1:4])
    runs, tests, rerun_diff = build(results_dir, paper_dir)
    for name, recs in runs.items():
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for r in recs:
            (d / f"fold{r['fold']}.json").write_text(json.dumps(r, default=float))
    sanitise(out_dir)
    (out_dir / 'foldwise.json').write_text(json.dumps(
        dict(summary={n: summary(r) for n, r in runs.items()}, tests=tests, max_abs_fold_auc_diff_rerun=rerun_diff), indent=1))
    (out_dir / 'foldwise.md').write_text(markdown(runs, tests, rerun_diff))
    print(markdown(runs, tests, rerun_diff))
