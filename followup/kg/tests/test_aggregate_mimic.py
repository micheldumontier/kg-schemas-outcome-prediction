import json
from aggregate_mimic import mimic_macros


def write(tmp, run, aucs):
    d = tmp / f'MIMIC24_{run}'
    d.mkdir(parents=True)
    for k, a in enumerate(aucs):
        (d / f'fold{k}.json').write_text(json.dumps(dict(auc_weighted=a, acc=0.6, f1_backhome=0.7, f1_rehab=0.5, f1_death=0.5)))


def test_macros_have_means_and_paired_comparisons(tmp_path):
    write(tmp_path, 'GBM', [0.80 + 0.002 * k for k in range(10)])
    write(tmp_path, 'caresm_ts', [0.79 + 0.002 * k for k in range(10)])
    tex = mimic_macros(tmp_path, runs={'GBM': 'GBM', 'caresm_ts': 'CARE'}, pairs=[('GBM', 'caresm_ts', 'GBMvsCARE')])
    assert '\\newcommand{\\maucGBM}{0.809\\,$\\pm$\\,0.006}' in tex
    assert '\\newcommand{\\mdiffGBMvsCARE}{+0.010}' in tex
    assert '\\newcommand{\\mpGBMvsCARE}{' in tex


def test_incomplete_run_is_refused(tmp_path):
    import pytest
    write(tmp_path, 'GBM', [0.8] * 9)
    with pytest.raises(ValueError, match='9 folds'):
        mimic_macros(tmp_path, runs={'GBM': 'GBM'}, pairs=[])


def test_other_landmarks_and_seeds_get_tagged_macros(tmp_path):
    for run, base in [('GBM', 0.80), ('sphn_ts', 0.78)]:
        d = tmp_path / f'MIMIC48_{run}'
        d.mkdir()
        for k in range(10):
            (d / f'fold{k}.json').write_text(json.dumps(dict(auc_weighted=base + 0.002 * k, acc=0.6)))
    tex = mimic_macros(tmp_path, runs={'GBM': 'GBM', 'sphn_ts': 'SPHN'}, pairs=[('GBM', 'sphn_ts', 'GBMvsSPHN')],
                       prefix='MIMIC48', tag='Hfortyeight')
    assert '\\newcommand{\\maucGBMHfortyeight}{0.809\\,$\\pm$\\,0.006}' in tex
    assert '\\newcommand{\\mdiffGBMvsSPHNHfortyeight}{+0.020}' in tex


def test_negative_differences_use_a_typographic_minus():
    from aggregate_mimic import signed
    assert signed(-0.0031) == '$-$0.003' and signed(0.017) == '+0.017'
