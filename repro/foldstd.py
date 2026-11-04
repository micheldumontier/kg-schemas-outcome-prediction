"""Fold-wise standardisation of graph literals (opt-in runner option --foldwise_std).

A literal entity looks like "<value>"^^<http://nvasc.org/dt/<datatype>>. Its z-score uses the mean and population SD
(np.std, SD 0 falls back to 1) of the distinct literal entities of the same datatype that belong to the training fold:
those that are the object (original triple direction) of a triple whose subject is a training patient or a node
specific to a training patient (IRI /p<i>/... for patient i, or the patient IRI synth_patient_<i> itself).
Non-literal nodes get 0. With every patient as training patient the result equals the global z-score of
followup/kg/build_graphs.py:to_processed, provided each literal hangs off a patient-specific node.
"""
import re
import numpy as np

LIT = re.compile(r'^"?([^"]*)"?\^\^<http://nvasc\.org/dt/([^>]+)>$')   # pandas drops the opening quote of the entity file
OWNER = re.compile(r'^<http://nvasc\.org/(?:p(\d+)/.*|synth_patient_(\d+))>$')


def parse_literals(names):
    """Return {datatype: (node indices, float values)}; unparsable values are skipped, as in to_processed."""
    by_dt = {}
    for idx, n in enumerate(names):
        m = LIT.match(str(n))
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            by_dt.setdefault(m.group(2), ([], []))
            by_dt[m.group(2)][0].append(idx)
            by_dt[m.group(2)][1].append(v)
    return {dt: (np.array(i, dtype=np.int64), np.array(v)) for dt, (i, v) in by_dt.items()}


def node_owner(names):
    """Patient index owning each node (-1 for shared nodes)."""
    owner = np.full(len(names), -1, dtype=np.int64)
    for idx, n in enumerate(names):
        m = OWNER.match(str(n))
        if m:
            owner[idx] = int(m.group(1) if m.group(1) is not None else m.group(2))
    return owner


def foldwise_numeric(num_nodes, literals, owner, triples, train_patients):
    """triples: int array [n, 3] (subject, relation, object) in the original direction.
    Returns (float array [num_nodes], number of training literals per datatype)."""
    in_train = np.isin(owner[triples[:, 0]], np.asarray(train_patients)) & (owner[triples[:, 0]] >= 0)
    train_objects = np.zeros(num_nodes, dtype=bool)
    train_objects[triples[in_train, 2]] = True
    out = np.zeros(num_nodes)
    counts = {}
    for dt, (idx, vals) in literals.items():
        sel = train_objects[idx]
        counts[dt] = int(sel.sum())
        if not sel.any():
            mu, sd = 0.0, 1.0
        else:
            mu, sd = vals[sel].mean(), vals[sel].std()
            sd = sd if sd > 0 else 1.0
        out[idx] = (vals - mu) / sd
    return out, counts
