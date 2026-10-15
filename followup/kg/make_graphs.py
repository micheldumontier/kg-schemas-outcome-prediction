"""Build graph variants from a landmark feature table on the A40 and check them before training.

Usage: python make_graphs.py <features_L24.parquet> <output dir>
Prints graph statistics and check results only. Patient-level files stay on the A40.
"""
import json, sys
from pathlib import Path
import joblib
import pandas as pd
from build_graphs import sphn_triples, caresm_triples, compile_caresm, time_before_triples, check_graph, to_processed

OUTCOME = {'home': 0, 'rehab': 1, 'death': 2}
# variant name -> (builder, layers the model will use)
VARIANTS = {
    'sphn_ts':     (lambda recs: [t for i, r in enumerate(recs) for t in sphn_triples(r, i)], 3),
    'sphn_tsr':    (lambda recs: (lambda tr: tr + time_before_triples(tr))([t for i, r in enumerate(recs) for t in sphn_triples(r, i)]), 3),
    'caresm_ts':   (lambda recs: [t for i, r in enumerate(recs) for t in caresm_triples(r, i)], 4),
    'caresm_comp': (lambda recs: compile_caresm([t for i, r in enumerate(recs) for t in caresm_triples(r, i)]), 3),
}


def main(features_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(features_path).reset_index(drop=True)
    recs = f.drop(columns=['hadm_id']).to_dict('records')
    joblib.dump([OUTCOME[o] for o in f['outcome']], out / 'outcomes.joblib')
    for name, (build, layers) in VARIANTS.items():
        tr = build(recs)
        problems = check_graph(tr, layers=layers, n_patients=len(recs))
        info = to_processed(tr, out / name, n_patients=len(recs))
        status = 'ok' if not problems and info['literals_expected'] == info['literals_with_value'] else 'FAILED'
        print(json.dumps({'variant': name, 'layers': layers, 'check': status, 'problems': problems[:3], **info}), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
