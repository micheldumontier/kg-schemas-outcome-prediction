import numpy as np
from embed_labels import to_init_matrix


def test_init_matrix_places_reduced_unit_scale_vectors_at_entity_rows():
    rng = np.random.default_rng(0)
    ids = [3, 7, 9] + list(range(20, 60))                 # 43 labelled nodes out of 100
    vecs = rng.normal(size=(len(ids), 768))
    m = to_init_matrix(ids, vecs, n_entities=100, dim=32)
    assert m.shape == (100, 32)
    assert np.allclose(m[[0, 1, 2, 99]], 0)                # unlabelled rows stay zero
    labelled = m[ids]
    assert np.allclose(labelled.mean(0), 0, atol=1e-6) and np.allclose(labelled.std(0), 1, atol=1e-6)


def test_global_scale_keeps_the_variance_profile_at_unit_overall_scale():
    from embed_labels import reduce
    z = reduce(np.random.default_rng(1).normal(size=(80, 768)), 32, 'global')
    assert np.isclose(z.std(), 1) and z[:, 0].std() > z[:, -1].std()


def test_shuffle_permutes_vectors_among_labelled_rows_only():
    from embed_labels import shuffle_rows
    m = np.zeros((10, 2)); ids = [2, 5, 7]
    m[ids] = [[1, 1], [2, 2], [3, 3]]
    s = shuffle_rows(m, ids, seed=0)
    assert sorted(map(tuple, s[ids])) == [(1, 1), (2, 2), (3, 3)]
    assert np.allclose(np.delete(s, ids, 0), 0) and not np.allclose(s, m)
