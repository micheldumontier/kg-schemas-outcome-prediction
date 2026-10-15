from pathlib import Path
import joblib
import pytest
from tabular_baselines import load_table, MODELS

CSV = '../neurovasc/data/syn_data.csv'
needs_data = pytest.mark.skipif(not Path(CSV).exists() or not Path('graphs/caresm_10000_outcomes.joblib').exists(),
                                reason='needs the TeamHeKA/neurovasc clone in ../neurovasc and the generated CARE-SM graph')


@needs_data
def test_tabular_uses_graph_patient_order():
    X, y, cols = load_table(CSV)
    assert list(y) == joblib.load('graphs/caresm_10000_outcomes.joblib')
    assert X.shape == (10000, 30) and 'output' not in cols and 'outcome' not in cols


def test_models_are_the_papers_baselines():
    assert set(MODELS) == {'LR', 'RF', 'NN', 'AGE', 'AGE_RF', 'EVENTS_RF'}
    nn = MODELS['NN']().steps[-1][1]
    assert nn.hidden_layer_sizes == (100, 50, 10) and nn.activation == 'tanh'
    assert MODELS['RF']().n_estimators == 100


@needs_data
def test_events_only_model_uses_the_eight_event_columns():
    X, y, cols = load_table(CSV)
    m = MODELS['EVENTS_RF']().fit(X[:200], y[:200])
    assert m.steps[0][1].transform(X[:3]).shape == (3, 8)


@needs_data
def test_cli_runs_a_model_subset_with_suffix(tmp_path):
    import subprocess, sys
    r = subprocess.run([sys.executable, '-W', 'ignore', 'tabular_baselines.py', CSV, '--models', 'AGE',
                        '--suffix', '_X', '--results_dir', str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / 'AGE_X').glob('fold*.json'))) == 10
    assert not (tmp_path / 'LR_X').exists()
