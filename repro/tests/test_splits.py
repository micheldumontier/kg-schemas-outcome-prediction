import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from splits import paper_splits

def test_paper_splits_match_original_protocol():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2], size=1000, p=[0.44, 0.43, 0.13])
    got = paper_splits(y)
    assert len(got) == 10
    for (tr, va, te), (non_test, test) in zip(got, StratifiedKFold(10, shuffle=True, random_state=77).split(y, y)):
        exp_tr, exp_va, _, _ = train_test_split(non_test, y[non_test], test_size=1/9, random_state=77)
        assert np.array_equal(te, test) and np.array_equal(tr, exp_tr) and np.array_equal(va, exp_va)
        assert len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te)
