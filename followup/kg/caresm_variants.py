"""Matched CARE-SM variants for a controlled comparison of patient shortcuts.

All four variants share one base: caresm_triples, except that an event time is linked with SIO "has start time"
(proc SIO_000680 tnode, tnode has-value literal) instead of the locally minted nvasc:hasTimePoint, where tnode is
the patient-specific node <http://nvasc.org/p<i>/<event>_start>. Hop counts are those of caresm_triples.
- caresm_a:        the base, no shortcuts;
- caresm_sio:      A plus (patient, SIO_000008 "has attribute", output) for every attribute output;
- caresm_feat:     A plus (patient, nvasc:compiled/has_<feature>, output) for every attribute output (named as in
                   compile_caresm, attributes only: no event-output or time shortcuts);
- caresm_sio_part: B plus (patient, SIO_000062 "is participant in", process) and (process, rdf:type, nvasc:event/<e>)
                   for every event process.
Attribute outputs are those typed nvasc:feature/... (gender and every numeric field); event outputs are typed
nvasc:event/....

Run on the GPU machine from the repository root, with the environment of repro/requirements.txt.
  python followup/kg/caresm_variants.py build <features_L24.parquet> ~/mimic-iv/derived/caresm_variants/graphs_L24 \
      ~/mimic-iv/derived/graphs_L24/outcomes.joblib
      Writes the four graphs in the runner's format and prints statistics, check results and hop maxima only. The
      reference outcomes are those of the template graphs (make_graphs.py), so the variants share their patients.
  python followup/kg/caresm_variants.py queue <seeds, e.g. 0 or 1,2> <parallel>
      Runs every variant at 3 and 4 layers with the unchanged runner repro/run_rgcn.py and the flags of mimic_runs.py,
      writing per-fold results to ~/mimic-iv/derived/caresm_variants/results (sanitise them with
      aggregate_caresm_variants.py before they leave the secure machine).
"""
import collections, json, os, re, subprocess, sys, time
from pathlib import Path
import joblib
from build_graphs import caresm_triples, check_graph, hop_distances, upstream, to_processed, patient_iri, nv, SIO, TYPE, NV

HAS_START = f'<{SIO}SIO_000680>'
HAS_ATTRIBUTE = f'<{SIO}SIO_000008>'
PARTICIPANT_IN = f'<{SIO}SIO_000062>'
HAS_VALUE = f'<{SIO}has-value>'
HOURS = f'^^<{NV}dt/hours>'
OUTCOME = {'home': 0, 'rehab': 1, 'death': 2}
LAYERS = (3, 4)
CHECK_LAYERS = 4

HOME = Path.home()
DERIVED = HOME / 'mimic-iv/derived/caresm_variants'
GRAPHS = DERIVED / 'graphs_L24'
RESULTS = DERIVED / 'results'
LOGS = DERIVED / 'logs'
REPRO = Path(__file__).resolve().parents[2] / 'repro'
PYTHON = Path(sys.executable)
ZERO = r'synth_patient_|/p[0-9]+/|\^\^'


def base_triples(rec, i):
    """caresm_triples with the event time linked through SIO has-start-time and has-value."""
    ctx_to_start = {}
    out = []
    for h, p, o in caresm_triples(rec, i):
        if p == nv('hasTimePoint'):
            ctx_to_start[o] = o.replace('_context>', '_start>')
            out.append((h, HAS_START, ctx_to_start[o]))
        elif p == f'<{SIO}SIO_000680>' and h in ctx_to_start:
            out.append((ctx_to_start[h], HAS_VALUE, o))
        else:
            out.append((h, p, o))
    return out


def _outputs(triples):
    """(patient, process, output, kind, class local name) for every output, in triple order; kind is feature or event."""
    s = lambda local: f'<{SIO}{local}>'
    role_of = {h: o for (h, p, o) in triples if p == s('has-role')}
    procs = collections.defaultdict(list)
    for h, p, o in triples:
        if p == s('is-realized-in'):
            procs[h].append(o)
    out_of = {h: o for (h, p, o) in triples if p == s('has-output')}
    outputs, cls_of = set(out_of.values()), {}
    for h, p, o in triples:
        if p == TYPE and h in outputs:
            for kind in ('feature', 'event'):
                if o.startswith(f'<{NV}{kind}/'):
                    cls_of[h] = (kind, o[len(f'<{NV}{kind}/'):-1])
    rows = []
    for patient, role in role_of.items():
        for proc in procs[role]:
            out = out_of.get(proc)
            if out in cls_of:
                rows.append((patient, proc, out) + cls_of[out])
    return rows


def sio_attribute_shortcuts(triples):
    return [(pt, HAS_ATTRIBUTE, out) for (pt, _, out, kind, _) in _outputs(triples) if kind == 'feature']


def feature_shortcuts(triples):
    return [(pt, nv(f'compiled/has_{name}'), out) for (pt, _, out, kind, name) in _outputs(triples) if kind == 'feature']


def participation(triples):
    added = []
    for pt, proc, _, kind, name in _outputs(triples):
        if kind == 'event':
            added += [(pt, PARTICIPANT_IN, proc), (proc, TYPE, nv(f'event/{name}'))]
    return added


VARIANTS = {
    'caresm_a':        lambda tr: [],
    'caresm_sio':      sio_attribute_shortcuts,
    'caresm_feat':     feature_shortcuts,
    'caresm_sio_part': lambda tr: sio_attribute_shortcuts(tr) + participation(tr),
}


def build(name, recs):
    base = [t for i, r in enumerate(recs) for t in base_triples(r, i)]
    return base + VARIANTS[name](base)


def max_hops(triples, patient_ids):
    """Largest message-passing distance from a patient to its own attribute values and to its own event times."""
    up = upstream(triples)
    own = collections.defaultdict(set)
    for h, _, o in triples:
        m = re.search(r'/p(\d+)/', h)
        if m and o.startswith('"') and '^^' in o:
            own[int(m.group(1))].add(o)
    best = {'attribute_values': 0, 'event_times': 0}
    for i in patient_ids:
        d = hop_distances(triples, patient_iri(i), up=up)
        for o in own[i]:
            key = 'event_times' if o.endswith(HOURS) else 'attribute_values'
            best[key] = max(best[key], d.get(o, float('inf')))
    return best


def write_variants(recs, outcomes, out, reference_outcomes):
    """Write outcomes and the four graphs; refuse if the outcomes differ from the reference (template graphs_L24)."""
    ref = list(joblib.load(reference_outcomes))
    if len(ref) != len(outcomes) or [int(x) for x in ref] != [int(x) for x in outcomes]:
        raise ValueError(f'outcomes differ from the reference {reference_outcomes}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(list(outcomes), out / 'outcomes.joblib')
    rows = []
    for name in VARIANTS:
        tr = build(name, recs)
        problems = check_graph(tr, layers=CHECK_LAYERS, n_patients=len(recs))
        hops = max_hops(tr, range(len(recs)))
        info = to_processed(tr, out / name, n_patients=len(recs))
        ok = not problems and info['literals_expected'] == info['literals_with_value']
        rows.append({'variant': name, 'check_layers': CHECK_LAYERS, 'check': 'ok' if ok else 'FAILED',
                     'problems': problems[:3], 'n_problems': len(problems), 'max_hops': hops, **info})
    return rows


def run_name(seed, variant, layers):
    return f'MIMIC24_{variant}_L{layers}' if seed == 0 else f'MIMIC24s{seed}_{variant}_L{layers}'


def command(seed, variant, layers, n_patients):
    """The CARE-SM invocation of followup/kg/mimic_runs.py with new inputs, depth and results directory."""
    return [str(PYTHON), '-W', 'ignore', str(REPRO / 'run_rgcn.py'), '--run', run_name(seed, variant, layers),
            '--prefix', str(GRAPHS / variant), '--outcomes', str(GRAPHS / 'outcomes.joblib'),
            '--num_patients', str(n_patients), '--layers', str(layers), '--emb_std', '1.0', '--zero_emb_regex', ZERO,
            '--epochs', '2000', '--device', 'cuda', '--seed', str(seed), '--results_dir', str(RESULTS)]


def queue(seeds, parallel):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / 'queue.pid').write_text(f'{os.getpid()}\n')
    n = len(joblib.load(GRAPHS / 'outcomes.joblib'))
    jobs = [(s, v, L) for s in seeds for L in sorted(LAYERS, reverse=True) for v in VARIANTS]
    active = {}
    while jobs or active:
        while jobs and len(active) < parallel:
            j = jobs.pop(0)
            name = run_name(*j)
            log = open(LOGS / f'{name}.log', 'a')
            active[name] = subprocess.Popen(command(*j, n_patients=n), cwd=REPRO, stdout=log,
                                            stderr=subprocess.STDOUT)
            print(f'{time.strftime("%H:%M:%S")} started {name} pid={active[name].pid}', flush=True)
        for name, p in list(active.items()):
            if p.poll() is not None:
                print(f'{time.strftime("%H:%M:%S")} finished {name} exit={p.returncode}', flush=True)
                del active[name]
        time.sleep(15)
    print('all done', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        import pandas as pd
        f = pd.read_parquet(sys.argv[2]).reset_index(drop=True)
        recs = f.drop(columns=['hadm_id']).to_dict('records')
        for row in write_variants(recs, [OUTCOME[o] for o in f['outcome']], sys.argv[3], sys.argv[4]):
            print(json.dumps(row), flush=True)
    elif sys.argv[1] == 'queue':
        queue([int(s) for s in sys.argv[2].split(',')], int(sys.argv[3]))
