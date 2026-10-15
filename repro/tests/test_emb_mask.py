from emb_mask import zero_mask


def test_patient_specific_nodes_and_typed_literals_are_masked():
    names = ['<http://nvasc.org/synth_patient_0>', '<http://nvasc.org/p0/age>', '"61.0"^^<http://nvasc.org/dt/age>',
             '<http://nvasc.org/code/nimodipine>', '"nimodipine"', '<http://semanticscience.org/resource/Role>']
    m = zero_mask(names, r'synth_patient_|/p\d+/|\^\^')
    assert list(m) == [True, True, True, False, False, False]


def test_no_pattern_masks_nothing():
    assert not zero_mask(['<a>', '"b"'], None).any()
