import json
import numpy as np
from aggregate_arms import summarise


def write(tmp, arm, seed, aucs):
    d = tmp / f'MIMIC24s{seed}_arm_{arm}'
    d.mkdir()
    for k, a in enumerate(aucs):
        (d / f'fold{k}.json').write_text(json.dumps({'auc_weighted': a}))


def test_arms_are_averaged_over_seeds_and_compared_with_random_by_fold(tmp_path):
    for s in (0, 1):
        write(tmp_path, 'random', s, [0.80 + 0.01 * k for k in range(10)])
        write(tmp_path, 'sapbert', s, [0.81 + 0.01 * k + 0.002 * s * (k % 2) for k in range(10)])
    write(tmp_path, 'hier', 0, [0.8] * 9)                      # incomplete arm is skipped
    rows = {r[0]: r for r in summarise(tmp_path, seeds=(0, 1))}
    assert set(rows) == {'random', 'sapbert'}
    assert np.isclose(rows['sapbert'][4], 0.0105) and rows['sapbert'][5] < 0.001
    assert rows['random'][4] is None


def test_macros_report_each_arm_and_the_largest_difference(tmp_path):
    from aggregate_arms import arm_macros, ARMS
    for arm in ARMS:
        for s in (0, 1, 2):
            write(tmp_path, arm, s, [0.80 + 0.01 * k + (0.003 if arm == 'owl2vec' and k % 2 else 0) for k in range(10)])
    tex = arm_macros(tmp_path)
    assert '\\newcommand{\\maucArmRandom}{0.845\\,$\\pm$\\,0.030}' in tex
    assert '\\newcommand{\\mdiffArmOwl}{+0.002}' in tex and '\\newcommand{\\mArmMaxDiff}{0.002}' in tex
