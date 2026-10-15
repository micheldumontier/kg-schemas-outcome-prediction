import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib


def tiny_graph(tmp):
    n = 40
    ents = [f'<http://nvasc.org/synth_patient_{i}>' for i in range(n)] + ['<x>']
    pd.DataFrame({'id': range(len(ents)), 'e': ents}).to_csv(tmp / 'g_entities.tsv', sep='\t', header=False, index=False)
    pd.DataFrame({'h': range(n), 'r': [0] * n, 't': [n] * n}).to_csv(tmp / 'g_triples.tsv', sep='\t', header=False, index=False)
    np.save(tmp / 'g_numeric.npy', np.zeros((len(ents), 1)))
    joblib.dump([i % 3 for i in range(n)], tmp / 'y.joblib')


def run(tmp, extra=()):
    cmd = [sys.executable, 'run_rgcn.py', '--prefix', str(tmp / 'g'), '--outcomes', str(tmp / 'y.joblib'),
           '--num_patients', '40', '--epochs', '2', '--run', 'T', '--results_dir', str(tmp / 'res'), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parents[1])


def test_runner_writes_one_json_per_fold(tmp_path):
    tiny_graph(tmp_path)
    r = run(tmp_path, ['--run_folds', '2'])
    assert r.returncode == 0, r.stderr
    files = sorted((tmp_path / 'res' / 'T').glob('fold*.json'))
    assert [f.name for f in files] == ['fold0.json', 'fold1.json']
    rec = json.loads(files[0].read_text())
    assert set(rec) >= {'run', 'fold', 'best_epoch', 'acc', 'auc_weighted', 'f1_backhome', 'f1_rehab', 'f1_death',
                        'f1_macro', 'f1_weighted', 'pred_class_counts'}


def test_runner_skips_completed_folds(tmp_path):
    tiny_graph(tmp_path)
    assert run(tmp_path, ['--run_folds', '1']).returncode == 0
    f0 = tmp_path / 'res' / 'T' / 'fold0.json'
    before = f0.stat().st_mtime_ns
    r = run(tmp_path, ['--run_folds', '2'])
    assert r.returncode == 0, r.stderr
    assert f0.stat().st_mtime_ns == before
    assert 'skip fold 0' in r.stdout
    assert (tmp_path / 'res' / 'T' / 'fold1.json').exists()


def test_runner_accepts_zero_embedding_regex(tmp_path):
    tiny_graph(tmp_path)
    r = run(tmp_path, ['--run_folds', '1', '--zero_emb_regex', r'synth_patient_'])
    assert r.returncode == 0, r.stderr
    assert 'zeroed embeddings: 40' in r.stdout


def test_runner_adds_inverse_relations_when_asked(tmp_path):
    tiny_graph(tmp_path)
    r = run(tmp_path, ['--run_folds', '1', '--both_directions'])
    assert r.returncode == 0, r.stderr
    assert 'num_relations=2' in r.stdout and 'edge_index=[2, 80]' in r.stdout


def test_runner_loads_initial_embeddings_for_listed_rows(tmp_path):
    tiny_graph(tmp_path)
    init = np.zeros((41, 32)); init[40] = 1.0                    # only the shared node '<x>' gets an embedding
    np.save(tmp_path / 'init.npy', init)
    r = run(tmp_path, ['--run_folds', '1', '--init_emb', str(tmp_path / 'init.npy')])
    assert r.returncode == 0, r.stderr
    assert 'initial embeddings loaded for 1 nodes' in r.stdout
