"""Verbatim port of notebooks/Graphs/data_preprocess_CARESM.ipynb.

Only changes vs. the notebook (marked CHANGED):
  - input/output paths are CLI arguments
  - `entity[entity.entity == v].id` (a linear scan per literal) replaced by an
    equivalent dict lookup, for speed; the resulting array is identical
  - diagnostic prints of how many literal rows each filter matched
"""
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

nt_path, out_prefix = sys.argv[1], sys.argv[2]  # CHANGED
FIX_LITERALS = '--fix_literals' in sys.argv  # CHANGED: opt-in fix, off by default

df = pd.read_csv(nt_path, sep=" ", header=None)
df.drop(columns=df.columns[-1], axis=1, inplace=True)
df.columns=['h', 'r', 't']

node_df = df.copy()

# Map id to entities and relations.
ent_to_id = {k: v for v, k in enumerate(set(node_df['h']).union(set(node_df['t'])), start=0)}
rel_to_id = {k: v for v, k in enumerate(set(node_df['r']), start=0)}

triples = node_df.copy()
triples["h"] = node_df.h.map(ent_to_id)
triples["t"] = node_df.t.map(ent_to_id)
triples["r"] = node_df.r.map(rel_to_id)

entity = pd.DataFrame({'id': list(ent_to_id.values()), 'entity': list(ent_to_id)})
relation = pd.DataFrame({'id': list(rel_to_id.values()), 'relation': list(rel_to_id)})

triples.to_csv(f'{out_prefix}_triples.tsv', sep='\t', index=False, header=None)
entity.to_csv(f'{out_prefix}_entities.tsv', sep='\t', index=False, header=None)
relation.to_csv(f'{out_prefix}_relations.tsv', sep='\t', index=False, header=None)
print("Triples / Entities / Relations saved.")
print(f"entities={len(entity)} relations={len(relation)} triples={len(triples)}")  # CHANGED

# Get literals.
numeric_df = node_df[node_df['r'] == '<http://sphn.org/has-value>'].copy()
if FIX_LITERALS:  # CHANGED: use the predicate CARE-SM actually emits, and keep the stripped value
    numeric_df = node_df[node_df['r'] == '<http://semanticscience.org/resource/has-value>'].copy()
    numeric_df['t_val'] = numeric_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#float>')
print("rows matched by numeric-literal filter <http://sphn.org/has-value>:", len(numeric_df))  # CHANGED
print("rows with <http://semanticscience.org/resource/has-value>:", int((node_df['r'] == '<http://semanticscience.org/resource/has-value>').sum()))  # CHANGED
numeric_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#float>')
numeric_values = pd.to_numeric(numeric_df['t_val'].values if FIX_LITERALS else numeric_df.t.values)
numeric_df['numeric'] = numeric_values
numeric_arr = np.zeros((len(entity), 1))
for i, v in numeric_df.t.items():
    num_id = ent_to_id[v]  # CHANGED (was entity[entity.entity == v].id)
    numeric_arr[num_id] = numeric_df.numeric.loc[i]

time_df = node_df[node_df['r'].str.contains('<http://semanticscience.org/resource/SIO_000680>|<http://semanticscience.org/resource/SIO_000681>')].copy()
print("timestamp literal rows:", len(time_df))  # CHANGED
time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
times = []
for i, t in time_df.sec.items():
    time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
    times.append(time.total_seconds())
time_df['sec'] = times

qt = QuantileTransformer(n_quantiles=10, random_state=0)
qt_times = qt.fit_transform(time_df.sec.values.reshape(-1,1))
time_df['sec'] = list(qt_times.reshape(-1,))
for i, v in time_df.t.items():
    num_id = ent_to_id[v]  # CHANGED (was entity[entity.entity == v].id)
    numeric_arr[num_id] = time_df.sec.loc[i]

np.save(f"{out_prefix}_numeric.npy", numeric_arr)
print("Literals saved. non-zero literal nodes:", int((numeric_arr != 0).sum()))  # CHANGED
