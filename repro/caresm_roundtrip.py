"""Is the CARE-SM graph lossless? Rebuild a per-patient feature table from triples alone and evaluate RF on it.

Path read for each patient: patient -has-role-> role -is-realized-in-> process -has-output-> output,
then the output's rdf:type value class (NCIT_<feature>_<value>) and, when present, its sio:has-value literal.
"""
import collections, json, re, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

P = '<http://semanticscience.org/resource/'
VALUE_CLASS = re.compile(r'<http://purl.obolibrary.org/obo/NCIT_([A-Za-z_0-9]+?)_(-?[0-9.]+)>')


def features_from_caresm(nt_path, num_patients):
    rows = collections.defaultdict(list)
    for line in open(nt_path):
        h, r, t = line.rstrip(' .\n').split(' ', 2)
        rows[r].append((h, t))
    role = dict(rows[P + 'has-role>'])
    realized = collections.defaultdict(list)
    for h, t in rows[P + 'is-realized-in>']:
        realized[h].append(t)
    output = dict(rows[P + 'has-output>'])
    types = collections.defaultdict(list)
    for h, t in rows['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']:
        types[h].append(t)
    value = {h: float(t.split('^^')[0].strip('"')) for h, t in rows[P + 'has-value>']}
    feats = []
    for i in range(num_patients):
        f = {}
        for proc in realized[role[f'<http://nvasc.org/synth_patient_{i}>']]:
            o = output[proc]
            for c in types[o]:
                m = VALUE_CLASS.match(c)
                if m:
                    f[m.group(1)] = value.get(o, float(m.group(2)))
                elif c.startswith('<http://purl.obolibrary.org/obo/NCIT_') and 'C18020' not in c:
                    f['event_' + c.rsplit('_', 1)[-1].rstrip('>')] = 1
        feats.append(f)
    return pd.DataFrame(feats).fillna(0)


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from splits import paper_splits
    from tabular_baselines import metrics
    nt_path, outcomes = sys.argv[1], sys.argv[2]
    y = np.asarray(joblib.load(outcomes))
    X = features_from_caresm(nt_path, len(y))
    print('reconstructed table', X.shape)
    out = Path('results') / 'ROUNDTRIP_RF'
    out.mkdir(parents=True, exist_ok=True)
    for k, (tr, va, te) in enumerate(paper_splits(y)):
        m = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(X.values[tr], y[tr])
        rec = dict(run='ROUNDTRIP_RF', fold=k, **metrics(y[te], m.predict_proba(X.values[te])))
        (out / f'fold{k}.json').write_text(json.dumps(rec, default=float))
    print('ROUNDTRIP_RF done')
