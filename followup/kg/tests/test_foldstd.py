import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'repro'))
from foldstd import parse_literals, node_owner, foldwise_numeric
from calibration import calibration_metrics

NV = 'http://nvasc.org/'


def graph():
    names = [f'<{NV}synth_patient_{i}>' for i in range(3)] + [f'<{NV}p{i}/age>' for i in range(3)]
    ages = ['10.0', '20.0', '60.0']
    names += [f'"{a}"^^<{NV}dt/age>' for a in ages] + ['"x"^^<' + NV + 'dt/age>', f'<{NV}shared>']
    triples = [(3 + i, 0, 6 + i) for i in range(3)] + [(i, 1, 3 + i) for i in range(3)]
    return names, np.array(triples)


def test_parse_and_owner():
    names, _ = graph()
    lit = parse_literals(names)
    assert list(lit) == ['age'] and list(lit['age'][1]) == [10.0, 20.0, 60.0]   # unparsable "x" skipped
    assert list(node_owner(names)) == [0, 1, 2, 0, 1, 2] + [-1] * 5


def test_all_patients_equals_global_zscore():
    names, tr = graph()
    z, c = foldwise_numeric(len(names), parse_literals(names), node_owner(names), tr, [0, 1, 2])
    v = np.array([10.0, 20.0, 60.0])
    assert np.allclose(z[6:9], (v - v.mean()) / v.std()) and c == {'age': 3}
    assert z[:6].tolist() == [0] * 6 and z[9] == 0


def test_training_only_statistics():
    names, tr = graph()
    z, c = foldwise_numeric(len(names), parse_literals(names), node_owner(names), tr, [0, 1])
    v = np.array([10.0, 20.0])
    assert c == {'age': 2}
    assert np.allclose(z[6:9], (np.array([10.0, 20.0, 60.0]) - v.mean()) / v.std())


def test_zero_sd_and_empty_fall_back():
    names, tr = graph()
    z, _ = foldwise_numeric(len(names), parse_literals(names), node_owner(names), tr, [0])
    assert np.allclose(z[6:9], [0, 10, 50])
    z, c = foldwise_numeric(len(names), parse_literals(names), node_owner(names), tr, [])
    assert c == {'age': 0} and np.allclose(z[6:9], [10, 20, 60])


def test_calibration_metrics():
    y = np.array([0, 1, 2, 2])
    perfect = np.eye(3)[y] * 0.97 + 0.01
    m = calibration_metrics(y, perfect)
    assert m['auc_macro'] == 1.0 and m['brier'] == pytest.approx(6e-4)
    uniform = np.full((4, 3), 1 / 3)
    m = calibration_metrics(y, uniform)
    assert m['brier'] == pytest.approx(2 / 3) and m['log_loss'] == pytest.approx(np.log(3))


def test_unquoted_entity_names_as_read_by_pandas():
    names, tr = graph()
    stripped = [n.replace('"', "") for n in names]
    a = parse_literals(names)['age'][1]
    b = parse_literals(stripped)['age'][1]
    assert list(a) == list(b)
