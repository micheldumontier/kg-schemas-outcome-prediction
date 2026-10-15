"""Build the real-SPHN pilot graph from a landmark feature table on the A40, check it, and write node labels.

Usage: python make_sphn_real.py <features.parquet> <output dir> [sphn_real|sphn_hier|sphn_hier_omop <omop_ancestors.tsv>]
Writes <out>/<variant>_{entities,relations,triples}.tsv, _numeric.npy, outcomes.joblib and <variant>_labels.tsv
(entity id and label for shared nodes only). sphn_hier adds the materialised code and class hierarchies;
sphn_hier_omop takes SNOMED CT, LOINC and ICD-10-PCS ancestors from an OMOP vocabulary export instead.
Prints graph statistics and check results only.
"""
import json, sys
from pathlib import Path
import joblib
import pandas as pd
from build_graphs import check_graph, to_processed
from build_sphn_real import sphn_real_triples, shared_labels, with_hierarchy
from terminology import load_extra

OUTCOME = {'home': 0, 'rehab': 1, 'death': 2}


def main(features_path, out, variant='sphn_real', omop=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(features_path).reset_index(drop=True)
    recs = f.drop(columns=['hadm_id']).to_dict('records')
    joblib.dump([OUTCOME[o] for o in f['outcome']], out / 'outcomes.joblib')
    per_patient = [sphn_real_triples(r, i) for i, r in enumerate(recs)]
    onto = []
    extra = load_extra(omop) if variant == 'sphn_hier_omop' else None
    if variant in ('sphn_hier', 'sphn_hier_omop'):
        per_patient, onto = with_hierarchy(per_patient, extra)
    tr = [t for p in per_patient for t in p] + onto
    problems = []                      # check each patient's own subgraph, with messages flowing both ways
    for i, p in enumerate(per_patient):
        problems += check_graph(p + onto + [(o, '<inverse>', h) for (h, _, o) in p + onto], layers=4, patient_ids=[i])
    info = to_processed(tr, out / variant, n_patients=len(recs))
    labels = shared_labels(tr, extra)
    ent = pd.read_csv(out / f'{variant}_entities.tsv', sep='\t', header=None, names=['id', 'e'], quoting=3)
    ids = dict(zip(ent.e, ent.id))
    with open(out / f'{variant}_labels.tsv', 'w') as fh:
        fh.writelines(f'{ids[n]}\t{lab}\n' for n, lab in sorted(labels.items(), key=lambda kv: ids[kv[0]]))
    status = 'ok' if not problems and info['literals_expected'] == info['literals_with_value'] else 'FAILED'
    print(json.dumps({'variant': variant, 'layers': 4, 'check': status, 'problems': problems[:3],
                      'shared_labels': len(labels), **info}), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
