import json
import numpy as np
from aggregate_foldwise import build, markdown, summary


def write(root, run, offset, with_cal=True):
    for k in range(10):
        r = dict(run=run, fold=k, auc_weighted=0.7 + offset + 0.01 * k, f1_macro=0.5 + offset, pred_class_counts=[1, 2, 3])
        if with_cal:
            r.update(auc_macro=0.7, brier=0.6 - offset, log_loss=1.0)
        (root / run).mkdir(parents=True, exist_ok=True)
        (root / run / f'fold{k}.json').write_text(json.dumps(r))


def test_build_and_markdown(tmp_path):
    res, paper = tmp_path / 'res', tmp_path / 'paper'
    for g in ('caresm_ts', 'sphn_ts', 'sphn_real'):
        write(res, f'MIMIC24_{g}_foldstd', 0.01)
        write(paper, f'MIMIC24_{g}', 0.0, with_cal=False)
    write(res, 'MIMIC24_caresm_ts_global_cal', 0.0)
    for t in ('GBM', 'LR', 'EVENTS_GBM', 'AGE_GBM'):
        write(res, f'MIMIC24_{t}_cal', 0.02)
    runs, tests, diff = build(res, paper)
    assert diff == 0.0
    t = [x for x in tests if x['comparison'] == 'sphn_ts: fold-wise - global (paper run)' and x['metric'] == 'auc_weighted'][0]
    assert abs(t['diff'] - 0.01) < 1e-12
    assert 'brier' in summary(runs['MIMIC24_GBM_cal'])
    assert 'Largest per-fold' in markdown(runs, tests, diff)
