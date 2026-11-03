"""Queue of R-GCN runs on the aneurysm-repair subset (paper flags), with GPU and RAM launch checks.

Run on the A40 from the repository's repro/ directory, with the Python of repro/requirements.txt.
  python ../followup/kg/cohort_runs.py <graphs dir> <results dir> <log dir> <parallel>
<graphs dir> holds the subset's graphs from make_graphs.py; run names are ANEU24_<variant>.
"""
import subprocess, sys, time
from pathlib import Path
import joblib

VARIANTS = {'caresm_ts': 4, 'sphn_ts': 3}
ZERO = r'synth_patient_|/p[0-9]+/|\^\^'


def command(graphs, results, variant):
    n = len(joblib.load(Path(graphs) / 'outcomes.joblib'))
    return [sys.executable, '-W', 'ignore', 'run_rgcn.py', '--run', f'ANEU24_{variant}',
            '--prefix', str(Path(graphs) / variant), '--outcomes', str(Path(graphs) / 'outcomes.joblib'),
            '--num_patients', str(n), '--layers', str(VARIANTS[variant]), '--emb_std', '1.0',
            '--zero_emb_regex', ZERO, '--epochs', '2000', '--device', 'cuda', '--seed', '0',
            '--results_dir', str(results)]


def resources_ok():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.split()[0]
    ram = subprocess.run(['free', '-g'], capture_output=True, text=True).stdout.splitlines()[1].split()[-1]
    return int(gpu) >= 8192 and int(ram) >= 8


if __name__ == '__main__':
    graphs, results, logs, parallel = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    logs.mkdir(parents=True, exist_ok=True)
    queue, active = list(VARIANTS), {}
    while queue or active:
        while queue and len(active) < parallel and resources_ok():
            v = queue.pop(0)
            active[v] = subprocess.Popen(command(graphs, results, v), stdout=open(logs / f'ANEU24_{v}.log', 'a'),
                                         stderr=subprocess.STDOUT)
            print(f'started ANEU24_{v}', flush=True)
            time.sleep(90)                      # let memory settle before the next launch check
        for v, p in list(active.items()):
            if p.poll() is not None:
                print(f'finished ANEU24_{v} exit={p.returncode}', flush=True)
                del active[v]
        time.sleep(15)
    print('all done', flush=True)
