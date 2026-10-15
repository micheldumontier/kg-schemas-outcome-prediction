import numpy as np
from owl2vec_embed import read_word2vec, init_matrix


def test_vectors_are_read_and_placed_at_covered_rows_only(tmp_path):
    p = tmp_path / 'e.txt'
    p.write_text('4 2\nhttp://a 1 2\nhttp://b 3 5\nword 0 0\nsun and moon 7 8\n')
    vec = read_word2vec(p)
    assert np.allclose(vec['http://b'], [3, 5]) and np.allclose(vec['sun and moon'], [7, 8])
    m, n = init_matrix(vec, [(1, 'http://a'), (4, 'http://b'), (6, 'http://missing')], 8, 2)
    assert n == 2 and np.allclose(m[[0, 2, 3, 5, 6, 7]], 0)
    assert np.isclose(m[[1, 4]].std(), 1) and np.allclose(m[[1, 4]].mean(0), 0)
