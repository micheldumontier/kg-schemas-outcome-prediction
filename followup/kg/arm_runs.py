"""Queue for the semantic-enrichment arms on the A40 (plan A). Reuses the reproduction runner run_rgcn.py.

Usage (from ~/neurovasc-repro): python arm_runs.py <arms> <seeds> <parallel>
e.g. python arm_runs.py random,shuffled,sapbert,owl2vec,hier 0,1,2 4
Every arm uses the real-SPHN 2026.1 graph at the 24 h landmark with four layers and messages in both directions. Arms
differ only in the shared nodes: their starting vectors (init files in the graph folder) or, for `hier`, the
materialised code and class hierarchies (`hier_omop`: SNOMED CT, LOINC and ICD-10-PCS ancestors from OMOP). Run names are MIMIC24s<seed>_arm_<arm>; results go to ~/mimic-iv/derived/results.
"""
import subprocess, sys, time
from pathlib import Path
import joblib

HOME = Path.home()
G = HOME / 'mimic-iv/derived/graphs_L24'
ZERO = r'synth_patient_|/p[0-9]+/|\^\^'
ARMS = {   # arm: (graph prefix, init file or None)
    'random': ('sphn_real', None),
    'shuffled': ('sphn_real', 'sphn_real_init_shuffled.npy'),
    'sapbert': ('sphn_real', 'sphn_real_init_sapbert.npy'),
    'owl2vec': ('sphn_real', 'sphn_real_init_owl2vec_s{seed}.npy'),
    'hier': ('sphn_hier', None),
    'hier_omop': ('sphn_hier_omop', None),
}


def run_name(arm, seed):
    return f'MIMIC24s{seed}_arm_{arm}'


def command(arm, seed):
    prefix, init = ARMS[arm]
    cmd = [sys.executable, '-W', 'ignore', 'run_rgcn.py', '--run', run_name(arm, seed),
           '--prefix', str(G / prefix), '--outcomes', str(G / 'outcomes.joblib'),
           '--num_patients', str(len(joblib.load(G / 'outcomes.joblib'))), '--layers', '4', '--both_directions',
           '--emb_std', '1.0', '--zero_emb_regex', ZERO, '--epochs', '2000', '--device', 'cuda', '--seed', str(seed),
           '--results_dir', str(HOME / 'mimic-iv/derived/results')]
    if init:
        path = G / init.format(seed=seed)
        assert path.exists(), f'missing {path}'
        cmd += ['--init_emb', str(path)]
    return cmd


if __name__ == '__main__':
    arms, seeds, parallel = sys.argv[1].split(','), [int(s) for s in sys.argv[2].split(',')], int(sys.argv[3])
    jobs = [(a, s) for s in seeds for a in arms]
    cmds = {run_name(*j): command(*j) for j in jobs}          # fail early on a missing init file
    Path('logs').mkdir(exist_ok=True)
    queue, active = list(cmds), {}
    while queue or active:
        while queue and len(active) < parallel:
            name = queue.pop(0)
            active[name] = subprocess.Popen(cmds[name], stdout=open(f'logs/{name}.log', 'a'), stderr=subprocess.STDOUT)
            print(f'started {name}', flush=True)
        for name, p in list(active.items()):
            if p.poll() is not None:
                print(f'finished {name} exit={p.returncode}', flush=True)
                del active[name]
        time.sleep(15)
    print('all done', flush=True)
