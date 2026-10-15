"""Which nodes start with a zero embedding: nodes whose name matches a regex (e.g. patient-specific nodes)."""
import re
import numpy as np


def zero_mask(names, pattern):
    if not pattern:
        return np.zeros(len(names), dtype=bool)
    rx = re.compile(pattern)
    return np.array([bool(rx.search(str(n))) for n in names])
