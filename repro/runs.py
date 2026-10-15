"""Registry of the paper's runs and a launcher that keeps the A40 busy with N concurrent runs."""
import subprocess, sys, time
from pathlib import Path

COMMON = ['--folds', '10', '--epochs', '2000', '--device', 'cuda', '--seed', '0']
RUNS = {
    'A':  ['--prefix', 'processed/caresm_ts',             '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '3'],
    'B':  ['--prefix', 'processed/caresm_ts',             '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '5'],
    'E':  ['--prefix', 'processed/caresm_ts_fixlit',      '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '4'],
    'F':  ['--prefix', 'processed/caresm_ts_fixlit',      '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '4', '--emb_std', '1.0'],
    'S':  ['--prefix', 'processed/sphn_ts',               '--outcomes', 'processed/sphn_outcomes.joblib',   '--layers', '3'],
    'G':  ['--prefix', 'processed/sphn_ts_ageAsDiag',     '--outcomes', 'processed/sphn_outcomes.joblib',   '--layers', '3'],
    'D2': ['--prefix', 'processed/caresm_ts_withoutcome', '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '3'],
    'E5': ['--prefix', 'processed/caresm_ts_fixlit',      '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '5'],
    'F5': ['--prefix', 'processed/caresm_ts_fixlit',      '--outcomes', 'processed/caresm_outcomes.joblib', '--layers', '5', '--emb_std', '1.0'],
}


def command(name):
    return [sys.executable, '-W', 'ignore', 'run_rgcn.py', '--run', name, *RUNS[name], *COMMON]


if __name__ == '__main__':
    names, parallel = sys.argv[1].split(','), int(sys.argv[2])
    Path('logs').mkdir(exist_ok=True)
    queue, active = list(names), {}
    while queue or active:
        while queue and len(active) < parallel:
            n = queue.pop(0)
            active[n] = subprocess.Popen(command(n), stdout=open(f'logs/{n}.log', 'a'), stderr=subprocess.STDOUT)
            print(f'started {n}', flush=True)
        for n, p in list(active.items()):
            if p.poll() is not None:
                print(f'finished {n} exit={p.returncode}', flush=True)
                del active[n]
        time.sleep(10)
