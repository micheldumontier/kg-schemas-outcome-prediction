"""Initial node embeddings from text labels (plan A, semantic enrichment).

Usage: python embed_labels.py <labels.tsv> <entities.tsv> <out.npy> [--model M] [--dim 32] [--scale global|per_dim]
                              [--shuffle SEED]
Embeds each shared node's label with a sentence-embedding model (default SapBERT, [CLS] pooling), reduces to `dim`
with PCA and rescales to the unit scale of the random initialisation it replaces, then writes a [num_entities, dim]
matrix whose unlabelled rows are zero. `--scale global` (default) divides by one overall standard deviation, which keeps
the PCA variance profile; `per_dim` standardises every dimension, as in the October 2026 pilot. `--shuffle SEED`
permutes the vectors across the labelled nodes (the shuffled control: same vectors, meaning broken).
"""
import argparse
import numpy as np


def reduce(vectors, dim, scale='global'):
    x = np.asarray(vectors, dtype=np.float64)
    x = x - x.mean(0)
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    z = x @ vt[:dim].T
    if scale == 'per_dim':
        return (z - z.mean(0)) / z.std(0)
    return z / z.std()


def to_init_matrix(ids, vectors, n_entities, dim, scale='per_dim'):
    out = np.zeros((n_entities, dim))
    out[np.asarray(ids)] = reduce(vectors, dim, scale)
    return out


def shuffle_rows(m, ids, seed):
    """Permute the rows of `m` among `ids`; a few nodes may keep their own vector, as in any random permutation."""
    ids = np.asarray(ids)
    out = m.copy()
    out[ids] = m[np.random.default_rng(seed).permutation(ids)]
    return out


def sapbert(texts, model):
    from sentence_transformers import SentenceTransformer, models
    enc = models.Transformer(model)                         # SapBERT is trained with [CLS] pooling
    st = SentenceTransformer(modules=[enc, models.Pooling(enc.get_word_embedding_dimension(), pooling_mode='cls')])
    return st.encode(texts, normalize_embeddings=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('labels_tsv'); ap.add_argument('entities_tsv'); ap.add_argument('out')
    ap.add_argument('--model', default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--scale', choices=['global', 'per_dim'], default='global')
    ap.add_argument('--shuffle', type=int, default=None)
    a = ap.parse_args()
    rows = [line.rstrip('\n').split('\t', 1) for line in open(a.labels_tsv)]
    ids, texts = [int(i) for i, _ in rows], [t for _, t in rows]
    n_entities = sum(1 for _ in open(a.entities_tsv))
    m = to_init_matrix(ids, sapbert(texts, a.model), n_entities, a.dim, a.scale)
    if a.shuffle is not None:
        m = shuffle_rows(m, ids, a.shuffle)
    np.save(a.out, m)
    print(f'embedded {len(texts)} labels with {a.model} (scale {a.scale}, shuffle {a.shuffle}); '
          f'wrote {a.out} [{n_entities} x {a.dim}]')
