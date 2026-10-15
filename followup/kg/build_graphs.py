"""Build SPHN-style, CARE-SM-style and compiled CARE-SM graphs from landmark feature records (plan C, task C5).

Triples are tuples of N-Triples terms. Design choices that avoid the defects found in the reproduction:
- numeric literals carry a per-feature datatype (<http://nvasc.org/dt/age>, .../dt/hours, ...), so values of
  different features are distinct nodes and can be standardised per feature;
- the CARE-SM value class is the feature (not the raw value), so classes are shared across patients;
- compile_caresm materialises feature-specific shortcut edges so every feature is two hops from the patient;
- check_graph refuses graphs with outcome edges or with features beyond the model's receptive field.
Patient IRIs follow the reproduction runner's convention (<http://nvasc.org/synth_patient_i>).
"""
import collections, math, re
from pathlib import Path
import numpy as np

NV, SPHN, SIO = 'http://nvasc.org/', 'http://sphn.org/', 'http://semanticscience.org/resource/'
TYPE = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
LABEL = '<http://www.w3.org/2000/01/rdf-schema#label>'
BEFORE = '<http://www.w3.org/2006/time#before>'
DRUGS = ['nimodipine', 'paracetamol', 'morphine', 'norepinephrine', 'vasopressor', 'milrinone']
PROCEDURES = ['evd', 'intubation', 'coiling', 'clipping', 'angioplasty']
EVENTS = DRUGS + PROCEDURES
NUMERIC = 'every non-event numeric field of the record (age, GCS, vital signs and labs)'
TIME_PREDICATES = {f'<{SPHN}hasStartDateTime>', f'<{SIO}SIO_000680>'}


def nv(path):
    return f'<{NV}{path}>'


def patient_iri(i):
    return nv(f'synth_patient_{i}')


def lit(value, datatype):
    return f'"{float(value)}"^^<{NV}dt/{datatype}>'


def _numeric_fields(rec):
    skip = set(EVENTS) | {'hadm_id', 'outcome', 'gender'}
    for k, v in rec.items():
        if k in skip or k.endswith('_h') or v is None:
            continue
        if isinstance(v, (int, float, np.floating, np.integer)) and not math.isnan(float(v)):
            yield k, float(v)


def _events(rec):
    for e in EVENTS:
        if rec.get(e) == 1 and rec.get(e + '_h') is not None and not math.isnan(float(rec[e + '_h'])):
            yield e, float(rec[e + '_h'])


def sphn_triples(rec, i):
    p, n = patient_iri(i), lambda s: nv(f'p{i}/{s}')
    t = [(p, nv('hasGender'), n('gender')), (n('gender'), f'<{SPHN}hasCode>', nv(f'code/gender_{rec.get("gender")}'))]
    for f, v in _numeric_fields(rec):
        if f == 'age':
            t += [(p, nv('hasAge'), n('age')), (n('age'), f'<{SPHN}hasQuantity>', n('age_q')), (n('age_q'), f'<{SPHN}hasValue>', lit(v, 'age'))]
        else:
            t += [(p, nv('hasDiagnosis'), n(f)), (n(f), LABEL, f'"{f}"'), (n(f), f'<{SPHN}hasQuantity>', n(f + '_q')),
                  (n(f + '_q'), f'<{SPHN}hasValue>', lit(v, f))]
    for e, h in _events(rec):
        link = nv('hasDrugAdministrationEvent') if e in DRUGS else nv('hasProcedure')
        t += [(p, link, n(e)), (n(e), LABEL, f'"{e}"'), (n(e), f'<{SPHN}hasCode>', nv(f'code/{e}')),
              (n(e), f'<{SPHN}hasStartDateTime>', lit(h, 'hours'))]
    return t


def caresm_triples(rec, i):
    p, n = patient_iri(i), lambda s: nv(f'p{i}/{s}')
    s = lambda local: f'<{SIO}{local}>'
    role = n('role')
    t = [(p, TYPE, s('Person')), (p, s('has-role'), role), (role, TYPE, s('Role'))]

    def output(name, cls):
        proc, out = n(name + '_process'), n(name + '_output')
        return proc, out, [(role, s('is-realized-in'), proc), (proc, TYPE, s('Process')), (proc, s('has-output'), out),
                           (out, TYPE, cls)]
    _, out, tr = output('gender', nv(f'feature/gender_{rec.get("gender")}'))
    t += tr
    for f, v in _numeric_fields(rec):
        _, out, tr = output(f, nv(f'feature/{f}'))
        t += tr + [(out, s('has-value'), lit(v, f))]
    for e, h in _events(rec):
        proc, out, tr = output(e, nv(f'event/{e}'))
        ctx = n(e + '_context')
        t += tr + [(proc, nv('hasTimePoint'), ctx), (ctx, s('SIO_000680'), lit(h, 'hours'))]
    return t


def compile_caresm(triples):
    """Add patient -> output shortcuts named after the output's feature or event class, and patient -> context for events."""
    s = lambda local: f'<{SIO}{local}>'
    role_of = {h: o for (h, p, o) in triples if p == s('has-role')}
    procs = collections.defaultdict(list)
    for h, p, o in triples:
        if p == s('is-realized-in'):
            procs[h].append(o)
    out_of = {h: o for (h, p, o) in triples if p == s('has-output')}
    ctx_of = {h: o for (h, p, o) in triples if p == nv('hasTimePoint')}
    cls_of = {h: o for (h, p, o) in triples if p == TYPE and (o.startswith(nv('feature/')[:-1]) or o.startswith(nv('event/')[:-1]))}
    added = []
    for patient, role in role_of.items():
        for proc in procs[role]:
            out = out_of.get(proc)
            name = cls_of.get(out, '').rsplit('/', 1)[-1].rstrip('>')
            if out and name:
                added.append((patient, nv(f'compiled/has_{name}'), out))
            if proc in ctx_of:
                added.append((patient, nv(f'compiled/has_{name}_time'), ctx_of[proc]))
    return triples + added


def time_before_triples(triples):
    """time:before between consecutive timed events of the same patient (by start time)."""
    by_patient = collections.defaultdict(list)
    for h, p, o in triples:
        if p in TIME_PREDICATES:
            m = re.search(r'/p(\d+)/', h)
            by_patient[m.group(1)].append((float(o.split('"')[1]), h))
    out = []
    for items in by_patient.values():
        items.sort()
        out += [(a[1], BEFORE, b[1]) for a, b in zip(items, items[1:]) if a[0] < b[0]]
    return out


def upstream(triples):
    up = collections.defaultdict(list)
    for h, _, o in triples:
        up[h].append(o)
    return up


def hop_distances(triples, start, max_hops=8, up=None):
    """Message-passing hops from start: a node receives from the objects of its own triples (edges reversed)."""
    up = up if up is not None else upstream(triples)
    dist, frontier = {start: 0}, [start]
    for d in range(1, max_hops + 1):
        nxt = []
        for node in frontier:
            for m in up[node]:
                if m not in dist:
                    dist[m] = d
                    nxt.append(m)
        frontier = nxt
    return dist


def check_graph(triples, layers, n_patients=None, patient_ids=None):
    problems = []
    if any('hasOutcome' in p or 'outcome_' in o for (_, p, o) in triples):
        problems.append('outcome edge present')
    up = upstream(triples)
    own_by_patient = collections.defaultdict(set)
    for h, _, o in triples:
        m = re.search(r'/p(\d+)/', h)
        if m and o.startswith('"') and '^^' in o:
            own_by_patient[int(m.group(1))].add(o)
    for i in (patient_ids if patient_ids is not None else range(n_patients)):
        d = hop_distances(triples, patient_iri(i), up=up)
        far = [o for o in own_by_patient[i] if o in d and d[o] > layers]     # only this patient's own values
        unreachable = [o for o in own_by_patient[i] if o not in d]
        if far or unreachable:
            problems.append(f'patient {i}: {len(far) + len(unreachable)} feature literals beyond {layers} hops')
    return problems


def to_processed(triples, prefix, n_patients):
    """Write <prefix>_entities.tsv, _relations.tsv, _triples.tsv, _numeric.npy in the reproduction runner's format.
    Typed numeric literals are standardised per datatype (z-score); other nodes get 0."""
    prefix = Path(prefix)
    ents, rels = {}, {}
    rows = []
    for h, p, o in triples:
        for x in (h, o):
            ents.setdefault(x, len(ents))
        rels.setdefault(p, len(rels))
        rows.append((ents[h], rels[p], ents[o]))
    for i in range(n_patients):
        assert patient_iri(i) in ents, f'patient {i} missing from graph'
    by_dt = collections.defaultdict(list)
    for e, idx in ents.items():
        m = re.match(r'^"([^"]*)"\^\^<' + re.escape(NV) + r'dt/([^>]+)>$', e)
        if m:
            try:
                by_dt[m.group(2)].append((idx, float(m.group(1))))
            except ValueError:
                pass                      # counted as expected but not ingested, so the check reports it
    numeric = np.zeros((len(ents), 1))
    for dt, items in by_dt.items():
        v = np.array([x for _, x in items])
        z = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
        for (idx, _), zi in zip(items, z):
            numeric[idx] = zi
    with open(f'{prefix}_entities.tsv', 'w') as f:
        f.writelines(f'{i}\t{e}\n' for e, i in ents.items())
    with open(f'{prefix}_relations.tsv', 'w') as f:
        f.writelines(f'{i}\t{r}\n' for r, i in rels.items())
    with open(f'{prefix}_triples.tsv', 'w') as f:
        f.writelines(f'{h}\t{r}\t{t}\n' for h, r, t in rows)
    np.save(f'{prefix}_numeric.npy', numeric)
    return {'entities': len(ents), 'relations': len(rels), 'triples': len(rows),
            'literals_expected': sum(1 for e in ents if f'^^<{NV}dt/' in e),       # independent of the value parser
            'literals_with_value': sum(len(v) for v in by_dt.values())}
