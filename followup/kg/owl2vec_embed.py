"""OWL2Vec* starting vectors for the shared nodes of the real-SPHN graph (plan A, semantic enrichment).

Usage (OWL2Vec* environment, Python 3.8):
    python owl2vec_embed.py <assembled.owl> <labels.tsv> <entities.tsv> <out.npy> <seed> [dim]
Trains OWL2Vec* (URI, literal and mixed documents; random walks of depth 3 over the projection built with its
structural reasoner) on the assembled ontology from build_ontology.py, then writes a [num_entities, dim] matrix with
the IRI vector of every shared node the ontology covers, globally rescaled to unit standard deviation. Shared nodes the
ontology does not cover (record label strings) stay zero, so they keep the runner's random initialisation.
Word2Vec runs on several threads, so a seed fixes the walks and initialisation but not the exact vectors.
"""
import os, subprocess, sys, tempfile
import numpy as np

CONFIG = """[BASIC]
ontology_file = {onto}
embedding_dir = {out}/

[DOCUMENT]
cache_dir = {cache}
ontology_projection = yes
walker = random
walk_depth = 3
URI_Doc = yes
Lit_Doc = yes
Mix_Doc = yes
Mix_Type = random

[MODEL]
embed_size = {dim}
iteration = 10
window = 5
min_count = 1
negative = 25
seed = {seed}
epoch = 100
"""


def read_word2vec(path):
    vec = {}
    with open(path) as f:
        dim = int(next(f).split()[1])
        for line in f:
            parts = line.rstrip('\n').split(' ')         # keys may contain spaces; the vector is the last `dim` fields
            vec[' '.join(parts[:-dim])] = np.array(parts[-dim:], dtype=np.float64)
    return vec


def init_matrix(vec, ids_iris, n_entities, dim):
    """Rows for covered IRIs, centred and divided by one overall standard deviation; returns matrix and coverage."""
    covered = [(i, vec[iri]) for i, iri in ids_iris if iri in vec]
    m = np.zeros((n_entities, dim))
    if covered:
        z = np.array([v for _, v in covered])
        z = z - z.mean(0)
        z = z / z.std()
        m[[i for i, _ in covered]] = z
    return m, len(covered)


def main(onto, labels_tsv, entities_tsv, out, seed, dim=32):
    shared = {int(l.split('\t', 1)[0]) for l in open(labels_tsv)}
    ent = [l.rstrip('\n').split('\t', 1) for l in open(entities_tsv)]
    ids_iris = [(int(i), e[1:-1]) for i, e in ent if int(i) in shared and e.startswith('<')]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, 'owl2vec.cfg')
        open(cfg, 'w').write(CONFIG.format(onto=onto, out=tmp, cache=os.path.join(tmp, 'cache'), dim=dim, seed=seed))
        subprocess.run([sys.executable, '-m', 'owl2vec_star.cli', 'standalone', '--config_file', cfg],
                       check=True, cwd=tmp, stdout=subprocess.DEVNULL)
        vec = read_word2vec(os.path.join(tmp, 'ontology.embeddings.txt'))
    m, n = init_matrix(vec, ids_iris, len(ent), dim)
    np.save(out, m)
    print('OWL2Vec* seed {}: vectors for {} of {} shared IRI nodes ({} shared nodes in all); wrote {}'.format(
        seed, n, len(ids_iris), len(shared), out))


if __name__ == '__main__':
    main(*sys.argv[1:5], int(sys.argv[5]), *(int(x) for x in sys.argv[6:7]))
