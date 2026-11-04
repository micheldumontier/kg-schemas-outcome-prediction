"""Summary of the matched CARE-SM variant runs: mean AUC per run and paired fold comparisons.

Usage: python followup/kg/aggregate_caresm_variants.py <variant results dir> <mimic24 results dir> <output dir>
e.g. ~/mimic-iv/derived/caresm_variants/results ~/mimic-iv/derived/results followup/results/caresm_variants
Reads every complete run (10 folds) of seeds 0, 1 and 2; incomplete runs are listed and skipped. The runner's folds
do not depend on the seed, so with several complete seeds each fold's AUC is averaged over seeds before the paired
t-test over the 10 folds; a comparison uses only the seeds complete on both sides. The references MIMIC24_caresm_ts
(4 layers) and MIMIC24_caresm_comp (3 layers) are read from the main MIMIC-IV results directory, read only.
Writes sanitised per-fold files of the variant runs (no pred_class_counts) and summary.md and summary.json.
Aggregates only: AUC means, SDs, differences and p-values.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

VARIANTS = ['caresm_a', 'caresm_sio', 'caresm_feat', 'caresm_sio_part']
LAYERS = (3, 4)
SEEDS = (0, 1, 2)
REFERENCES = {'caresm_ts': 4, 'caresm_comp': 3}
VARIANT_PAIRS = [('caresm_sio', 'caresm_a'), ('caresm_feat', 'caresm_a'), ('caresm_feat', 'caresm_sio'),
                 ('caresm_sio_part', 'caresm_sio')]
COUNT_FIELDS = ('pred_class_counts',)
LABEL = {'caresm_a': 'A', 'caresm_sio': 'B', 'caresm_feat': 'C', 'caresm_sio_part': 'D',
         'caresm_ts': 'CARE-SM (MIE)', 'caresm_comp': 'compiled CARE-SM (MIE)'}


def run_dir(seed, key):
    return f'MIMIC24_{key}' if seed == 0 else f'MIMIC24s{seed}_{key}'


def keys():
    return [f'{v}_L{L}' for L in LAYERS for v in VARIANTS] + list(REFERENCES)


def pairs():
    out = [(f'{a}_L{L}', f'{b}_L{L}') for L in LAYERS for a, b in VARIANT_PAIRS]
    out += [(f'{v}_L3', f'{v}_L4') for v in VARIANTS]
    out += [('caresm_comp', 'caresm_ts'), ('caresm_a_L4', 'caresm_ts'), ('caresm_feat_L3', 'caresm_comp')]
    return out


def read_folds(d):
    files = sorted(Path(d).glob('fold*.json'), key=lambda p: int(p.stem[4:]))
    return [json.loads(f.read_text()) for f in files]


def aggregate(results_dir, reference_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    auc, incomplete = {}, []                          # (key, seed) -> array of 10 fold AUCs
    for key in keys():
        src = Path(reference_dir if key in REFERENCES else results_dir)
        for s in SEEDS:
            d = src / run_dir(s, key)
            if not d.exists():
                continue
            folds = read_folds(d)
            if len(folds) != 10:
                incomplete.append(run_dir(s, key))
                continue
            auc[key, s] = np.array([f['auc_weighted'] for f in folds])
            if key not in REFERENCES:                 # sanitised copy of the variant runs
                (out_dir / run_dir(s, key)).mkdir(exist_ok=True)
                for k, f in enumerate(folds):
                    clean = {a: b for a, b in f.items() if a not in COUNT_FIELDS}
                    (out_dir / run_dir(s, key) / f'fold{k}.json').write_text(json.dumps(clean))
    seeds_of = {key: [s for s in SEEDS if (key, s) in auc] for key in keys()}
    runs = []
    for key in keys():
        layers = REFERENCES.get(key) or int(key.rsplit('_L', 1)[1])
        variant = key if key in REFERENCES else key.rsplit('_L', 1)[0]
        for s in seeds_of[key]:
            v = auc[key, s]
            runs.append({'run': run_dir(s, key), 'variant': LABEL[variant], 'layers': layers, 'seeds': [s],
                         'folds': len(v), 'auc_mean': float(v.mean()), 'auc_sd': float(v.std(ddof=1))})
        if len(seeds_of[key]) > 1:
            v = np.mean([auc[key, s] for s in seeds_of[key]], axis=0)
            runs.append({'run': f'{key} (mean of seeds {",".join(map(str, seeds_of[key]))})', 'variant': LABEL[variant],
                         'layers': layers, 'seeds': seeds_of[key], 'folds': len(v),
                         'auc_mean': float(v.mean()), 'auc_sd': float(v.std(ddof=1))})
    comparisons = []
    for a, b in pairs():
        common = [s for s in seeds_of[a] if s in seeds_of[b]]
        if not common:
            continue
        x = np.mean([auc[a, s] for s in common], axis=0)
        y = np.mean([auc[b, s] for s in common], axis=0)
        # no p-value when every fold differs by the same amount (zero variance of the differences)
        pv = None if np.std(x - y) < 1e-12 else float(stats.ttest_rel(x, y).pvalue)
        comparisons.append({'comparison': f'{a} vs {b}', 'seeds': common, 'mean_a': float(x.mean()),
                            'mean_b': float(y.mean()), 'diff': float((x - y).mean()),
                            'p': pv})
    summary = {'metric': 'auc_weighted', 'folds': 10, 'runs': runs, 'pairs': comparisons, 'incomplete': incomplete}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out_dir / 'summary.md').write_text(markdown(summary))
    return summary


def pvalue(p):
    return 'n/a' if p is None else '<0.001' if p < 0.001 else f'{p:.3f}'


def markdown(summary):
    lines = ['# Matched CARE-SM variants (MIMIC-IV SAH, 24-hour landmark)', '',
             'Generated by followup/kg/aggregate_caresm_variants.py; do not edit. Weighted one-vs-rest AUC over the '
             'runner\'s 10 folds. A: common base (SIO start time); B: A plus SIO has-attribute shortcuts; C: A plus '
             'per-feature shortcuts; D: B plus SIO participation and process typing. References from the MIE runs.',
             '', '| run | variant | layers | folds | AUC mean | AUC SD |', '|---|---|---|---|---|---|']
    for r in summary['runs']:
        lines.append(f'| {r["run"]} | {r["variant"]} | {r["layers"]} | {r["folds"]} | {r["auc_mean"]:.3f} | '
                     f'{r["auc_sd"]:.3f} |')
    lines += ['', 'Paired t-tests over the 10 folds (fold AUCs averaged over the listed seeds).', '',
              '| comparison | seeds | mean a | mean b | difference | p |', '|---|---|---|---|---|---|']
    for p in summary['pairs']:
        lines.append(f'| {p["comparison"]} | {",".join(map(str, p["seeds"]))} | {p["mean_a"]:.3f} | {p["mean_b"]:.3f} | '
                     f'{p["diff"]:+.3f} | {pvalue(p["p"])} |')
    if summary['incomplete']:
        lines += ['', 'Incomplete runs, skipped: ' + ', '.join(summary['incomplete']) + '.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    s = aggregate(*sys.argv[1:4])
    print('runs', len(s['runs']), 'pairs', len(s['pairs']), 'incomplete', len(s['incomplete']))
