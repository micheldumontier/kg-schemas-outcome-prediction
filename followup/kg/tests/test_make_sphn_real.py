"""End-to-end build of both real-SPHN variants on three synthetic records."""
import json
import numpy as np
import pandas as pd
import pytest
from make_sphn_real import main
from tests.test_build_sphn_real import REC


@pytest.mark.parametrize('variant', ['sphn_real', 'sphn_hier'])
def test_build_passes_checks_and_labels_shared_nodes(tmp_path, capsys, variant):
    recs = [dict(REC, hadm_id=1, outcome='home'), dict(REC, hadm_id=2, outcome='death', gender='M', age=80.0),
            dict(REC, hadm_id=3, outcome='rehab', coiling=1, coiling_h=7.0)]
    pd.DataFrame(recs).to_parquet(tmp_path / 'f.parquet')
    main(tmp_path / 'f.parquet', tmp_path / 'g', variant)
    info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert info['check'] == 'ok' and info['variant'] == variant
    labels = dict(l.rstrip('\n').split('\t') for l in open(tmp_path / 'g' / f'{variant}_labels.tsv'))
    assert 'nimodipine' in labels.values()
    assert ('calcium channel blockers' in labels.values()) == (variant == 'sphn_hier')


def test_omop_variant_uses_supplied_ancestors(tmp_path, capsys):
    from build_sphn_real import SNOMED
    omop = tmp_path / 'anc.tsv'
    omop.write_text(f'<{SNOMED}386554004>\t<{SNOMED}273249006>\tAssessment scales\t1\n'
                    f'<{SNOMED}386554004>\t<{SNOMED}254291000>\tStaging and scales\t2\n')
    recs = [dict(REC, hadm_id=1, outcome='home'), dict(REC, hadm_id=2, outcome='death')]
    pd.DataFrame(recs).to_parquet(tmp_path / 'f.parquet')
    main(tmp_path / 'f.parquet', tmp_path / 'g', 'sphn_hier_omop', omop)
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['check'] == 'ok'
    labels = set(l.rstrip('\n').split('\t')[1] for l in open(tmp_path / 'g' / 'sphn_hier_omop_labels.tsv'))
    assert {'Assessment scales', 'Staging and scales', 'calcium channel blockers'} <= labels
