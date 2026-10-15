"""How many RGCN layers does a patient node need before it can see each feature?

The runner reverses every triple (h, r, t) into an edge t -> h, so a node receives
messages from the *objects* of its own triples. A feature is visible to a k-layer
RGCN only if its informative node lies within k such hops upstream of the patient.
Informative nodes are those whose identity or literal value distinguishes patients:
feature-value class/code nodes and numeric/time literals.
"""
import sys, re, collections
import numpy as np
import pandas as pd

prefix, num_patients = sys.argv[1], int(sys.argv[2])
ent = pd.read_csv(f'{prefix}_entities.tsv', sep='\t', header=None, names=['id', 'e'])
tr = pd.read_csv(f'{prefix}_triples.tsv', sep='\t', header=None, names=['h', 'r', 't'])
rel = pd.read_csv(f'{prefix}_relations.tsv', sep='\t', header=None, names=['id', 'r'])
num = np.load(f'{prefix}_numeric.npy').ravel()
name = dict(zip(ent.id, ent.e))
rname = dict(zip(rel.id, rel.r))
pid = {e: i for i, e in name.items()}

feats = ['hospital_stay_length', 'gcs', 'nb_acte', 'age', 'gender', 'entry', 'entry_code', 'ica', 'ttt', 'ica_therapy',
         'fever', 'o2_clinic', 'o2', 'hta', 'hct', 'tabagisme', 'etOH', 'diabete', 'headache', 'instable', 'vasospasme', 'ivh']

def kind(n):
    s = str(name[n])
    m = re.search(r'(?:NCIT_|code_)(' + '|'.join(sorted(feats, key=len, reverse=True)) + r')_', s)
    if m:
        return 'class:' + m.group(1)
    if num[n] != 0:
        return 'literal(nonzero numeric input)'
    return None

up = collections.defaultdict(list)          # node -> upstream neighbours (objects of its triples)
for h, t in zip(tr.h.values, tr.t.values):
    up[h].append(t)

first_seen = collections.defaultdict(list)  # kind -> list of hop distances (per patient, min)
for i in range(0, num_patients, max(1, num_patients // 500)):   # sample 500 patients
    p = pid[f'<http://nvasc.org/synth_patient_{i}>']
    dist = {p: 0}; frontier = [p]; seen_k = {}
    for d in range(1, 7):
        nxt = []
        for n in frontier:
            for m in up[n]:
                if m not in dist:
                    dist[m] = d; nxt.append(m)
                    k = kind(m)
                    if k and k not in seen_k:
                        seen_k[k] = d
        frontier = nxt
    for k, d in seen_k.items():
        first_seen[k].append(d)

print(f'{prefix}: relations = {sorted(rname.values())}')
print(f'{"informative node kind":45s} min/median/max hops upstream of patient (500 sampled patients)')
for k in sorted(first_seen):
    v = first_seen[k]
    print(f'  {k:43s} {min(v)}/{int(np.median(v))}/{max(v)}   (reached for {len(v)}/500)')

# how many distinct class nodes per numeric feature? (unique per patient => carries no shared signal)
for f in ['age', 'hospital_stay_length', 'gcs', 'nb_acte', 'diabete']:
    cls = [e for e in name.values() if re.search(rf'(?:NCIT_|code_){f}_', str(e))]
    print(f'  distinct class/code nodes for {f:22s}: {len(cls)}  e.g. {cls[:2]}')
