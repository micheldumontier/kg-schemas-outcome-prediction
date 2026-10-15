"""The original paper's split: StratifiedKFold(10, shuffle, rs=77), then 1/9 of non-test for validation (rs=77)."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def paper_splits(y):
    y = np.asarray(y)
    out = []
    for non_test, test in StratifiedKFold(10, shuffle=True, random_state=77).split(y, y):
        train, val, _, _ = train_test_split(non_test, y[non_test], test_size=1/9, random_state=77)
        out.append((train, val, test))
    return out
