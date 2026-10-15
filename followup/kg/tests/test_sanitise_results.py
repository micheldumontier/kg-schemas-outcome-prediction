import json
from sanitise_results import sanitise


def test_only_summary_metrics_are_kept(tmp_path):
    f = tmp_path / 'MIMIC24_GBM' / 'fold0.json'
    f.parent.mkdir()
    f.write_text(json.dumps(dict(run='MIMIC24_GBM', fold=0, auc_weighted=0.8, acc=0.6, pred_class_counts=[69, 64, 6])))
    assert sanitise(tmp_path) == 1
    out = json.loads(f.read_text())
    assert 'pred_class_counts' not in out and out['auc_weighted'] == 0.8
    assert sanitise(tmp_path) == 0                       # idempotent
