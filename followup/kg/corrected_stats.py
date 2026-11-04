"""Paired comparisons over the 10 cross-validation folds with the Nadeau-Bengio corrected resampled t-test.

Usage: python corrected_stats.py <mimic24 results dir> <caresm_variants results dir> <output prefix> [synthetic results dir]
e.g. python followup/kg/corrected_stats.py followup/results/mimic24 followup/results/caresm_variants
  followup/results/corrected_stats repro/results, from the repository root.
Training sets of the folds overlap, so the plain paired t-test understates the variance of a mean fold difference.
The corrected test (Nadeau and Bengio, 2003) scales the variance by 1/k + n_test/n_train; with the paper's splits
(one tenth for testing, one ninth of the rest for validation) n_test/n_train = 0.1/0.8 = 0.125. Runs over several
seeds are averaged within each fold first, as in aggregate_arms.py. Writes <prefix>.md and <prefix>.json with the
mean difference, the plain and corrected p-values and the corrected 95% confidence interval (aggregates only).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

K, RATIO = 10, 0.1 / 0.8


def folds(results_dir, run):
    files = sorted(Path(results_dir, run).glob('fold*.json'), key=lambda p: int(p.stem[4:]))
    if len(files) != K:
        raise ValueError(f'{run} has {len(files)} folds, expected {K}')
    return np.array([json.loads(f.read_text())['auc_weighted'] for f in files])


def corrected(d):
    """Mean difference, plain p, corrected p and corrected 95% CI for per-fold differences d."""
    m, s2 = d.mean(), d.var(ddof=1)
    se_plain, se_corr = np.sqrt(s2 / K), np.sqrt((1 / K + RATIO) * s2)
    p_plain = 2 * stats.t.sf(abs(m) / se_plain, K - 1)
    p_corr = 2 * stats.t.sf(abs(m) / se_corr, K - 1)
    h = stats.t.ppf(0.975, K - 1) * se_corr
    return dict(diff=float(m), p=float(p_plain), p_corr=float(p_corr), ci=[float(m - h), float(m + h)])


def seeds_mean(results_dir, runs):
    return np.mean([folds(results_dir, r) for r in runs], axis=0)


def comparisons(mimic_dir, variants_dir):
    rows = []
    add = lambda label, a, b: rows.append(dict(comparison=label, **corrected(a - b)))
    f = lambda run: folds(mimic_dir, run)
    for L in (24, 48, 72):
        P = f'MIMIC{L}'
        for a, b in [('caresm_ts', 'sphn_ts'), ('caresm_comp', 'caresm_ts'), ('sphn_tsr', 'sphn_ts'),
                     ('GBM', 'caresm_ts'), ('GBM', 'sphn_ts'), ('GBM', 'sphn_real'), ('sphn_real', 'sphn_ts')]:
            add(f'{L} h: {a} - {b}', f(f'{P}_{a}'), f(f'{P}_{b}'))
    for a, b in [('sphn_ts_bidir', 'sphn_ts'), ('sphn_real', 'sphn_ts_bidir'), ('sphn_real_text', 'sphn_real')]:
        add(f'24 h: {a} - {b}', f(f'MIMIC24_{a}'), f(f'MIMIC24_{b}'))
    seeds = lambda run: seeds_mean(mimic_dir, [f'MIMIC24_{run}', f'MIMIC24s1_{run}', f'MIMIC24s2_{run}'])
    for a, b in [('caresm_ts', 'sphn_ts'), ('caresm_comp', 'caresm_ts'), ('sphn_tsr', 'sphn_ts')]:
        add(f'24 h, 3 seeds: {a} - {b}', seeds(a), seeds(b))
    arm = lambda name: seeds_mean(mimic_dir, [f'MIMIC24s{s}_arm_{name}' for s in (0, 1, 2)])
    for name in ('shuffled', 'sapbert', 'owl2vec', 'hier', 'hier_omop'):
        add(f'24 h, 3 seeds: arm {name} - arm random', arm(name), arm('random'))
    var = lambda v, L: seeds_mean(variants_dir, [f'MIMIC24{s}_caresm_{v}_L{L}' for s in ('', 's1', 's2')])
    names = {'A': 'a', 'B': 'sio', 'C': 'feat', 'D': 'sio_part'}
    for L in (3, 4):
        for a, b in [('B', 'A'), ('C', 'A'), ('C', 'B'), ('D', 'B')]:
            add(f'24 h, 3 seeds, {L} layers: variant {a} - {b}', var(names[a], L), var(names[b], L))
    return rows


SYNTHETIC = [('S', 'G'), ('LR', 'S'), ('ROUNDTRIP_RF', 'S'), ('AGE_RF', 'S'), ('E', 'A'), ('E5', 'E'), ('F', 'E'),
             ('S', 'F'), ('S', 'F5')]


def synthetic(results_dir):
    """Synthetic runs of repro/runs.py (same split proportions, so the same correction applies)."""
    return [dict(comparison=f'synthetic: {a} - {b}', **corrected(folds(results_dir, a) - folds(results_dir, b)))
            for a, b in SYNTHETIC]


def markdown(rows):
    out = ['| Comparison | Difference | p (plain) | p (corrected) | 95% CI (corrected) |', '|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['comparison']} | {r['diff']:+.3f} | {r['p']:.3f} | {r['p_corr']:.3f} | "
                   f"{r['ci'][0]:+.3f} to {r['ci'][1]:+.3f} |")
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    rows = comparisons(sys.argv[1], sys.argv[2]) + (synthetic(sys.argv[4]) if len(sys.argv) > 4 else [])
    Path(sys.argv[3] + '.json').write_text(json.dumps(rows, indent=1))
    Path(sys.argv[3] + '.md').write_text(markdown(rows))
    print(markdown(rows))
