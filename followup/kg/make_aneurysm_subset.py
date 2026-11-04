"""Restrict the 24 h landmark table to admissions with aneurysm repair (coiling or clipping) at any time.

Usage: python make_aneurysm_subset.py <features_L24.parquet> <events.parquet> <output parquet>
Both inputs come from followup/mimic/build_features.py. Row order of the feature table is kept. Prints aggregate
counts only (counts below 11 suppressed). The subset then goes through make_graphs.py (graphs), tabular_mimic.py with
run prefix ANEU24 (tabular models), cohort_runs.py (graph runs) and aggregate_cohort.py (summary).
Selecting on repair during the admission uses information after the landmark, so this is a sensitivity analysis of
the cohort definition, not a deployable model.
"""
import sys
import pandas as pd

REPAIR = ('coiling', 'clipping')


def suppress(n):
    return int(n) if n >= 11 else '<11'


def select_repair(features, events):
    """Return (subset, repaired_before_landmark) where subset keeps the rows of `features` whose hadm_id has a repair event."""
    ids = set(events.loc[events['event'].isin(REPAIR), 'hadm_id'])
    keep = features['hadm_id'].isin(ids)
    subset = features[keep].reset_index(drop=True)
    early = int(((subset[list(REPAIR)] > 0).any(axis=1)).sum())
    return subset, early


def main(features_path, events_path, out):
    f = pd.read_parquet(features_path)
    sub, early = select_repair(f, pd.read_parquet(events_path))
    sub.to_parquet(out)
    print('full cohort', suppress(len(f)), '| subset', suppress(len(sub)))
    print('outcome classes', {k: suppress(v) for k, v in sub['outcome'].value_counts().items()})
    print('repair before 24 h', suppress(early))


if __name__ == '__main__':
    main(*sys.argv[1:4])
