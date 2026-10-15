"""Build the project page (docs/index.html) from site/template.html and the per-fold results in this repository.

Usage: python site/build.py            (from the repository root; needs numpy and scipy)
Every {{name}} in the template is replaced by a value computed here from repro/results and followup/results; the build
fails if a placeholder has no value or a run is incomplete, so the page cannot drift from the results.
"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SYN, MIMIC = ROOT / 'repro/results', ROOT / 'followup/results/mimic24'

# Aggregate cohort sizes from the cohort report (counts of at least 11 only)
COHORT = {'n24': '1,388', 'n48': '1,287', 'n72': '1,190', 'home24': '560', 'rehab24': '530', 'death24': '298'}


def folds(path, metric='auc_weighted'):
    files = sorted(Path(path).glob('fold*.json'), key=lambda p: int(p.stem[4:]))
    if len(files) != 10:
        raise ValueError(f'{path} has {len(files)} folds, expected 10')
    return np.array([json.loads(f.read_text())[metric] for f in files])


def ms(v):
    return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'


def signed(x):
    return f'{x:+.3f}'.replace('-', '−')


def paired(a, b):
    """Mean difference a − b over folds, its 95% confidence interval and the paired t-test p-value."""
    d = a - b
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - half, d.mean() + half, stats.ttest_rel(a, b).pvalue


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}' if p < 0.01 else f'{p:.2f}'


def values():
    v = dict(COHORT)
    for run in ['A', 'B', 'E', 'E5', 'F', 'F5', 'S', 'G', 'D2', 'LR', 'RF', 'NN', 'AGE_RF', 'EVENTS_RF',
                'LR_SEPT2024', 'RF_SEPT2024', 'NN_SEPT2024', 'ROUNDTRIP_RF']:
        v[f'syn_{run}'] = ms(folds(SYN / run))
        v[f'synacc_{run}'] = ms(folds(SYN / run, 'acc'))
    runs = ['GBM', 'LR', 'caresm_ts', 'caresm_comp', 'sphn_ts', 'sphn_tsr', 'EVENTS_GBM', 'AGE_GBM']
    for L in (24, 48, 72):
        for r in runs:
            v[f'm{L}_{r}'] = ms(folds(MIMIC / f'MIMIC{L}_{r}'))
    for r in ['sphn_ts_bidir', 'sphn_real']:
        v[f'm24_{r}'] = ms(folds(MIMIC / f'MIMIC24_{r}'))
    v['macc24_GBM'] = ms(folds(MIMIC / 'MIMIC24_GBM', 'acc'))

    rows = []        # CARE-SM minus template SPHN, every landmark and seed
    for label, prefix in [('24 h, seed 0', 'MIMIC24'), ('48 h, seed 0', 'MIMIC48'), ('72 h, seed 0', 'MIMIC72'),
                          ('24 h, seed 1', 'MIMIC24s1'), ('24 h, seed 2', 'MIMIC24s2')]:
        d, lo, hi, p = paired(folds(MIMIC / f'{prefix}_caresm_ts'), folds(MIMIC / f'{prefix}_sphn_ts'))
        rows.append(f'<tr><td>{label}</td><td class="num">{signed(d)}</td><td class="num">{signed(lo)} to {signed(hi)}</td>'
                    f'<td class="num">{fmt_p(p)}</td></tr>')
    v['rows_care_sphn'] = '\n'.join(rows)

    pairs = [('Compiled CARE-SM minus CARE-SM', 'caresm_comp', 'caresm_ts'),
             ('SPHN with time relations minus SPHN', 'sphn_tsr', 'sphn_ts'),
             ('Gradient boosting minus CARE-SM', 'GBM', 'caresm_ts'),
             ('Gradient boosting minus template SPHN', 'GBM', 'sphn_ts'),
             ('Template SPHN, both directions minus one direction', 'sphn_ts_bidir', 'sphn_ts'),
             ('Released SPHN 2026.1 minus template SPHN, both directions', 'sphn_real', 'sphn_ts_bidir'),
             ('Gradient boosting minus released SPHN 2026.1', 'GBM', 'sphn_real')]
    rows = []
    for label, a, b in pairs:
        d, lo, hi, p = paired(folds(MIMIC / f'MIMIC24_{a}'), folds(MIMIC / f'MIMIC24_{b}'))
        rows.append(f'<tr><td>{label}</td><td class="num">{signed(d)}</td><td class="num">{signed(lo)} to {signed(hi)}</td>'
                    f'<td class="num">{fmt_p(p)}</td></tr>')
    v['rows_pairs24'] = '\n'.join(rows)

    arms = [('random', 'Random vectors (reference)'), ('shuffled', 'SapBERT label embeddings, shuffled across nodes'),
            ('sapbert', 'SapBERT label embeddings'), ('owl2vec', 'OWL2Vec* ontology embeddings'),
            ('hier', 'ATC and ICD-10-PCS hierarchies as edges'),
            ('hier_omop', 'OMOP hierarchies as edges (adds SNOMED CT and LOINC)')]
    per_arm = {a: np.array([folds(MIMIC / f'MIMIC24s{s}_arm_{a}') for s in (0, 1, 2)]).mean(0) for a, _ in arms}
    rows, forest = [], []
    for a, label in arms:
        if a == 'random':
            rows.append(f'<tr><td>{label}</td><td class="num">{ms(per_arm[a])}</td><td></td><td></td><td></td></tr>')
            continue
        d, lo, hi, p = paired(per_arm[a], per_arm['random'])
        rows.append(f'<tr><td>{label}</td><td class="num">{ms(per_arm[a])}</td><td class="num">{signed(d)}</td>'
                    f'<td class="num">{signed(lo)} to {signed(hi)}</td><td class="num">{fmt_p(p)}</td></tr>')
        forest.append((label, d, lo, hi))
    v['rows_arms'] = '\n'.join(rows)
    v['forest_arms'] = forest_svg(forest)
    return v


def forest_svg(items, span=0.012):
    """Differences from the reference with 95% confidence intervals, drawn to one scale from −span to +span."""
    W, left, right, top, row = 720, 330, 30, 34, 34
    H = top + row * len(items) + 30
    x = lambda d: left + (d + span) / (2 * span) * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Difference in AUC from random starting vectors with 95% '
           f'confidence intervals; every interval includes zero.">']
    for t in (-0.01, -0.005, 0, 0.005, 0.01):
        cls = 'zero' if t == 0 else 'grid'
        out.append(f'<line class="{cls}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 8}" y2="{H - 26}"/>')
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{signed(t) if t else "0"}</text>')
    out.append(f'<text class="tick" x="{x(0):.1f}" y="{top - 16}" text-anchor="middle">difference in AUC</text>')
    for i, (label, d, lo, hi) in enumerate(items):
        y = top + row * i + row / 2
        out.append(f'<text class="lab" x="{left - 14}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
        out.append(f'<line class="ci" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<circle class="pt" cx="{x(d):.1f}" cy="{y:.1f}" r="5"/>')
    out.append('</svg>')
    return '\n'.join(out)


def build(template, out):
    text = template.read_text()
    v = values()
    missing = sorted(set(re.findall(r'\{\{(\w+)\}\}', text)) - set(v))
    if missing:
        raise KeyError(f'no value for {missing}')
    out.parent.mkdir(exist_ok=True)
    out.write_text(re.sub(r'\{\{(\w+)\}\}', lambda m: v[m.group(1)], text))
    return out


if __name__ == '__main__':
    print('wrote', build(ROOT / 'site/template.html', ROOT / 'docs/index.html'))
