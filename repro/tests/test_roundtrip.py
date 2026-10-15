from pathlib import Path
import joblib, pandas as pd
import pytest
from caresm_roundtrip import features_from_caresm

CSV = '../neurovasc/data/syn_data.csv'
pytestmark = pytest.mark.skipif(not Path(CSV).exists() or not Path('graphs/caresm_50.nt').exists(),
                                reason='needs the TeamHeKA/neurovasc clone in ../neurovasc and graphs/caresm_50.nt')


def test_caresm_graph_returns_each_patients_values():
    feats = features_from_caresm('graphs/caresm_50.nt', 50)
    df = pd.read_csv(CSV, index_col=0).sample(frac=1, random_state=42).reset_index(drop=True)
    assert len(feats) == 50
    assert list(feats.age) == [round(a) for a in df.age[:50]]            # has-value literal is the rounded age
    assert list(feats.diabete) == list(df.diabete[:50].astype(float))     # categorical value read from the class IRI
    assert {c for c in feats.columns if c.startswith('event_')} <= {f'event_{c}' for c in
            ['C08CA06', 'N02BE01', 'C01CA03', 'C01CE02', 'N02AA01', '00P6X0Z', 'Z98.6', '0BH17EZ']}
