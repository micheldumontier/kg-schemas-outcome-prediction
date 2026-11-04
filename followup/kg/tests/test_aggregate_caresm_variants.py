"""Tests for the CARE-SM variant summary, on synthetic fold files only."""
import json
import numpy as np
from scipy import stats
from aggregate_caresm_variants import aggregate, VARIANT_PAIRS, run_dir

BASE = {'caresm_a': 0.75, 'caresm_sio': 0.78, 'caresm_feat': 0.79, 'caresm_sio_part': 0.80}


def write(root, name, aucs):
    d = root / name
    d.mkdir(parents=True)
    for k, a in enumerate(aucs):
        (d / f'fold{k}.json').write_text(json.dumps(dict(run=name, fold=k, auc_weighted=a, acc=0.6,
                                                         pred_class_counts=[60, 50, 28])))


def aucs(base, seed, layers):
    return [base + 0.004 * k + 0.003 * seed - 0.002 * (layers == 3) * (k % 4) + 0.001 * ((k * 7 + seed) % 3)
            + 0.001 * ((k * round(base * 100)) % 5) for k in range(10)]


def setup(tmp, seeds=(0,), incomplete=None):
    new, ref = tmp / 'new', tmp / 'ref'
    for s in seeds:
        for v, b in BASE.items():
            for L in (3, 4):
                vals = aucs(b, s, L)
                write(new, run_dir(s, f'{v}_L{L}'), vals[:9] if (s, v, L) == incomplete else vals)
        write(ref, run_dir(s, 'caresm_ts'), aucs(0.76, s, 4))
        write(ref, run_dir(s, 'caresm_comp'), aucs(0.79, s, 3))
    return new, ref


def test_seed_zero_table_pairs_and_sanitised_copies(tmp_path):
    new, ref = setup(tmp_path)
    out = tmp_path / 'out'
    summary = aggregate(new, ref, out)
    row = next(r for r in summary['runs'] if r['run'] == 'MIMIC24_caresm_sio_L4')
    v = np.array(aucs(0.78, 0, 4))
    assert np.isclose(row['auc_mean'], v.mean()) and np.isclose(row['auc_sd'], v.std(ddof=1)) and row['folds'] == 10
    assert {r['run'] for r in summary['runs']} >= {'MIMIC24_caresm_ts', 'MIMIC24_caresm_comp'}
    names = {p['comparison'] for p in summary['pairs']}
    for L in (3, 4):
        for a, b in VARIANT_PAIRS:
            assert f'{a}_L{L} vs {b}_L{L}' in names
    for v_ in BASE:
        assert f'{v_}_L3 vs {v_}_L4' in names
    p = next(p for p in summary['pairs'] if p['comparison'] == 'caresm_sio_L4 vs caresm_a_L4')
    x, y = np.array(aucs(0.78, 0, 4)), np.array(aucs(0.75, 0, 4))
    assert np.isclose(p['diff'], (x - y).mean()) and np.isclose(p['p'], stats.ttest_rel(x, y).pvalue)
    assert p['seeds'] == [0]
    copied = list(out.rglob('fold*.json'))
    assert len(copied) == 80
    assert all('pred_class_counts' not in json.loads(f.read_text()) for f in copied)
    md = (out / 'summary.md').read_text()
    assert 'MIMIC24_caresm_feat_L3' in md and 'pred_class_counts' not in md
    assert json.loads((out / 'summary.json').read_text())['pairs'] == summary['pairs']


def test_identical_fold_differences_give_no_p_value(tmp_path):
    new, ref = tmp_path / 'new', tmp_path / 'ref'
    for v in BASE:
        for L in (3, 4):
            write(new, run_dir(0, f'{v}_L{L}'), [0.75 + 0.01 * k + 0.002 * (L == 4) for k in range(10)])
    summary = aggregate(new, ref, tmp_path / 'out')
    p = next(p for p in summary['pairs'] if p['comparison'] == 'caresm_a_L3 vs caresm_a_L4')
    assert p['p'] is None and np.isclose(p['diff'], -0.002)
    assert 'n/a' in (tmp_path / 'out' / 'summary.md').read_text()


def test_complete_seeds_are_averaged_per_fold_and_incomplete_runs_skipped(tmp_path):
    new, ref = setup(tmp_path, seeds=(0, 1, 2), incomplete=(2, 'caresm_a', 4))
    summary = aggregate(new, ref, tmp_path / 'out')
    assert 'MIMIC24s2_caresm_a_L4' in summary['incomplete']
    assert not any(r['run'] == 'MIMIC24s2_caresm_a_L4' for r in summary['runs'])
    mean_row = next(r for r in summary['runs'] if r['run'] == 'caresm_sio_L4 (mean of seeds 0,1,2)')
    avg = np.mean([aucs(0.78, s, 4) for s in (0, 1, 2)], axis=0)
    assert np.isclose(mean_row['auc_mean'], avg.mean())
    p = next(p for p in summary['pairs'] if p['comparison'] == 'caresm_sio_L4 vs caresm_a_L4')
    assert p['seeds'] == [0, 1]                         # seed 2 of A at 4 layers is incomplete
    x = np.mean([aucs(0.78, s, 4) for s in (0, 1)], axis=0)
    y = np.mean([aucs(0.75, s, 4) for s in (0, 1)], axis=0)
    assert np.isclose(p['p'], stats.ttest_rel(x, y).pvalue)
    r = next(p for p in summary['pairs'] if p['comparison'] == 'caresm_comp vs caresm_ts')
    assert r['seeds'] == [0, 1, 2]
