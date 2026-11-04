import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_aneurysm_subset import select_repair, suppress


def test_select_repair_keeps_order_and_counts_early():
    f = pd.DataFrame({'hadm_id': [5, 3, 9, 1], 'outcome': ['home'] * 4,
                      'coiling': [1, 0, 0, 0], 'clipping': [0, 0, 0, 0]})
    e = pd.DataFrame({'hadm_id': [5, 9, 3, 7], 'event': ['coiling', 'clipping', 'evd', 'coiling'],
                      'time': pd.Timestamp('2020-01-01'), 'date_only': True})
    sub, early = select_repair(f, e)
    assert list(sub['hadm_id']) == [5, 9]
    assert early == 1


def test_suppress():
    assert suppress(10) == '<11' and suppress(11) == 11
