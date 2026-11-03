import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aggregate_cohort as ac


def write(root, run, auc, f1):
    (root / run).mkdir(parents=True)
    for k in range(10):
        (root / run / f'fold{k}.json').write_text(json.dumps(dict(auc_weighted=auc + k * 0.01, f1_macro=f1)))


def test_summarise_and_markdown(tmp_path):
    sub, full = tmp_path / 's', tmp_path / 'f'
    for m in ac.MODELS:
        write(sub, f'ANEU24_{m}', 0.7 if m != 'GBM' else 0.8, 0.5)
        write(full, f'MIMIC24_{m}', 0.6, 0.4)
    res = ac.summarise(sub, full)
    t = [t for t in res['tests'] if t['comparison'] == 'GBM - sphn_ts' and t['metric'] == 'auc_weighted'][0]
    assert abs(t['diff'] - 0.1) < 1e-9
    assert abs(res['models']['LR']['auc_weighted']['full_mean'] - 0.645) < 1e-9
    md = ac.markdown(res, 432, dict(home=195, rehab=5, death=59))
    assert 'rehab <11' in md and 'GBM - sphn_ts' in md


def test_suppress():
    assert ac.suppress(10) == '<11' and ac.suppress(11) == 11
