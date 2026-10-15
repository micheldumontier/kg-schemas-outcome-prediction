"""Queue of MIMIC-IV graph runs on the A40 (reuses the reproduction runner run_rgcn.py).

Usage (on the A40, from ~/neurovasc-repro): python mimic_runs.py <spec> <parallel>
<spec> is a comma-separated list of LANDMARK:SEED, e.g. 48:0,72:0,24:1,24:2. Each expands to the four graph variants.
Run names are MIMIC<L>_<variant> for seed 0 and MIMIC<L>s<seed>_<variant> otherwise, written to ~/mimic-iv/derived/results.
"""
import subprocess, sys, time
from pathlib import Path
import joblib

HOME = Path.home()
VARIANTS = {'sphn_ts': 3, 'sphn_tsr': 3, 'caresm_ts': 4, 'caresm_comp': 3}
ZERO = r'synth_patient_|/p[0-9]+/|\^\^'


def run_name(landmark, seed, variant):
    return f'MIMIC{landmark}_{variant}' if seed == 0 else f'MIMIC{landmark}s{seed}_{variant}'


def command(landmark, seed, variant):
    g = HOME / f'mimic-iv/derived/graphs_L{landmark}'
    n = len(joblib.load(g / 'outcomes.joblib'))
    return [sys.executable, '-W', 'ignore', 'run_rgcn.py', '--run', run_name(landmark, seed, variant),
            '--prefix', str(g / variant), '--outcomes', str(g / 'outcomes.joblib'), '--num_patients', str(n),
            '--layers', str(VARIANTS[variant]), '--emb_std', '1.0', '--zero_emb_regex', ZERO,
            '--epochs', '2000', '--device', 'cuda', '--seed', str(seed),
            '--results_dir', str(HOME / 'mimic-iv/derived/results')]


if __name__ == '__main__':
    jobs = [(int(L), int(s), v) for item in sys.argv[1].split(',') for L, s in [item.split(':')] for v in VARIANTS]
    jobs.sort(key=lambda j: j[2] != 'caresm_comp')           # slowest variant first so the queue drains evenly
    parallel = int(sys.argv[2])
    Path('logs').mkdir(exist_ok=True)
    queue, active = list(jobs), {}
    while queue or active:
        while queue and len(active) < parallel:
            j = queue.pop(0)
            name = run_name(*j)
            active[name] = subprocess.Popen(command(*j), stdout=open(f'logs/{name}.log', 'a'), stderr=subprocess.STDOUT)
            print(f'started {name}', flush=True)
        for name, p in list(active.items()):
            if p.poll() is not None:
                print(f'finished {name} exit={p.returncode}', flush=True)
                del active[name]
        time.sleep(15)
    print('all done', flush=True)
