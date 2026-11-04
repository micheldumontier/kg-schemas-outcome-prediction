import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "repro"))
import numpy as np
import pytest
from calibration_slopes import slope, citl, ece, pooled_loss, analyse, load_pooled, markdown, RUNS
from calibration import save_probs


def test_slope_recovers_overconfidence():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.05, 0.95, 20000)
    ind = (rng.random(20000) < true).astype(int)
    lg = np.log(true / (1 - true))
    assert slope(ind, true) == pytest.approx(1, abs=0.08)
    over = 1 / (1 + np.exp(-2 * lg))                  # logit doubled, so the true slope is 0.5
    assert slope(ind, over) == pytest.approx(0.5, abs=0.06)


def test_citl_and_ece():
    ind = np.array([1, 0, 1, 0]); p = np.array([.9, .1, .9, .1])
    assert citl(ind, p) == pytest.approx(0.0)
    assert ece(ind, p, bins=2) == pytest.approx(0.1)
    assert citl(np.zeros(4, int), np.full(4, .3)) == pytest.approx(0.3)
    assert ece(np.zeros(4, int), np.full(4, .3), bins=2) == pytest.approx(0.3)


def test_pooled_loss_uniform():
    b, ll = pooled_loss(np.array([0, 1, 2]), np.full((3, 3), 1 / 3))
    assert b == pytest.approx(2 / 3) and ll == pytest.approx(np.log(3))


def test_pooling_and_analyse(tmp_path):
    rng = np.random.default_rng(0)
    n = 300; y = rng.integers(0, 3, n); prob = rng.dirichlet([1, 1, 1], n); perm = rng.permutation(n)
    for k in range(10):
        i = perm[k::10]
        save_probs(tmp_path, 'R', k, i, y[i], prob[i])
    y2, p2 = load_pooled(tmp_path, 'R')
    assert (y2 == y).all() and np.allclose(p2, prob)
    r = analyse(y2, p2, boot=20)
    assert r['n_patients'] == n and r['home']['slope']['ci'][0] <= r['home']['slope']['ci'][1]
    assert 'Calibration slope' in markdown({'MIMIC24_GBM_probs': r})


def test_duplicate_patient_rejected(tmp_path):
    for k in range(10):
        save_probs(tmp_path, 'R', k, [0, 1], [0, 1], np.full((2, 3), 1 / 3))
    with pytest.raises(AssertionError):
        load_pooled(tmp_path, 'R')
