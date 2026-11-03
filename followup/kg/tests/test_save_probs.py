import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'repro'))
from calibration import save_probs, platt_recalibrate


def test_save_probs_roundtrip(tmp_path):
    prob = np.array([[.2, .3, .5], [.6, .3, .1]])
    save_probs(tmp_path, 'RUN', 3, [7, 9], [2, 0], prob)
    z = np.load(tmp_path / 'RUN' / 'fold3.npz')
    assert z['idx'].tolist() == [7, 9] and z['y'].tolist() == [2, 0] and np.allclose(z['prob'], prob)
    assert [p.name for p in (tmp_path / 'RUN').iterdir()] == ['fold3.npz']


def test_platt_recalibration_shrinks_overconfidence():
    rng = np.random.default_rng(0)
    n = 3000
    true = rng.dirichlet([2, 2, 2], n)
    y = np.array([rng.choice(3, p=p) for p in true])
    over = true ** 3 / (true ** 3).sum(1, keepdims=True)         # overconfident version of the truth
    out = platt_recalibrate(over[:2000], y[:2000], over[2000:])
    brier = lambda p, t: ((p - np.eye(3)[t]) ** 2).sum(1).mean()
    assert np.allclose(out.sum(1), 1) and brier(out, y[2000:]) < brier(over[2000:], y[2000:])
