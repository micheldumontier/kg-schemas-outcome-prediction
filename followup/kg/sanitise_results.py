"""Remove per-fold counts of patients from MIMIC-IV result files before they leave the secure environment.

Usage: python sanitise_results.py <results dir>
Per-fold files keep summary metrics only (AUC, accuracy, F1, best epoch); counts of predicted patients per class
can fall below the small-cell threshold of 11 and are dropped.
"""
import json, sys
from pathlib import Path

COUNT_FIELDS = ('pred_class_counts',)


def sanitise(results_dir):
    changed = 0
    for f in Path(results_dir).rglob('fold*.json'):
        d = json.loads(f.read_text())
        if any(k in d for k in COUNT_FIELDS):
            f.write_text(json.dumps({k: v for k, v in d.items() if k not in COUNT_FIELDS}))
            changed += 1
    return changed


if __name__ == '__main__':
    print('sanitised', sanitise(sys.argv[1]), 'files')
