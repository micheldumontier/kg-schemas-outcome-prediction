"""Tests for SPHN, CARE-SM and compiled CARE-SM graph construction, on synthetic records only."""
import numpy as np
from build_graphs import (sphn_triples, caresm_triples, compile_caresm, time_before_triples, hop_distances,
                          to_processed, check_graph, patient_iri, NUMERIC, EVENTS)

REC = {'age': 61.0, 'gender': 'F', 'gcs_first': 9.0, 'sodium_min': 131.0,
       'nimodipine': 1, 'nimodipine_h': 5.0, 'evd': 1, 'evd_h': 2.0, 'coiling': 0, 'coiling_h': np.nan}


def literals(triples):
    return [t for (_, _, t) in triples if t.startswith('"')]


def test_sphn_record_has_age_values_and_only_events_that_happened():
    tr = sphn_triples(REC, 0)
    assert any(p.endswith('hasAge>') for (_, p, _) in tr)
    assert '"61.0"^^<http://nvasc.org/dt/age>' in literals(tr)
    assert '"5.0"^^<http://nvasc.org/dt/hours>' in literals(tr)          # nimodipine start, hours since admission
    assert not any('coiling' in t for t in literals(tr) + [o for (_, _, o) in tr])


def test_caresm_puts_features_four_hops_away_and_compiler_brings_them_to_two():
    tr = caresm_triples(REC, 0)
    d = hop_distances(tr, patient_iri(0))
    age = '"61.0"^^<http://nvasc.org/dt/age>'
    assert d[age] == 4
    d2 = hop_distances(compile_caresm(tr), patient_iri(0))
    assert d2[age] == 2


def test_time_relations_link_consecutive_events_in_time_order():
    tr = sphn_triples(REC, 0)
    before = time_before_triples(tr)
    assert len(before) == 1                                             # evd (2 h) before nimodipine (5 h)
    s, p, o = before[0]
    assert p.endswith('time#before>') and 'evd' in s and 'nimodipine' in o


def test_processed_numeric_values_are_standardised_per_datatype(tmp_path):
    recs = [dict(REC, age=a) for a in (40.0, 60.0, 80.0)]
    tr = [t for i, r in enumerate(recs) for t in sphn_triples(r, i)]
    info = to_processed(tr, tmp_path / 'g', n_patients=3)
    num = np.load(tmp_path / 'g_numeric.npy').ravel()
    ent = dict(line.rstrip('\n').split('\t')[::-1] for line in open(tmp_path / 'g_entities.tsv'))
    z = [num[int(ent[f'"{a}"^^<http://nvasc.org/dt/age>'])] for a in (40.0, 60.0, 80.0)]
    assert np.allclose(z, [-1.2247, 0.0, 1.2247], atol=1e-3)
    assert info['literals_expected'] == info['literals_with_value']


def test_check_graph_flags_unreachable_features_and_outcome_edges():
    tr = caresm_triples(REC, 0)
    problems = check_graph(tr, layers=3, n_patients=1)
    assert any('beyond' in p for p in problems)
    assert check_graph(compile_caresm(tr), layers=3, n_patients=1) == []
    leaky = tr + [(patient_iri(0), '<http://nvasc.org/hasOutcome>', '<http://nvasc.org/outcome_death>')]
    assert any('outcome' in p for p in check_graph(compile_caresm(leaky), layers=3, n_patients=1))


def test_literal_check_is_independent_of_the_parser(tmp_path):
    # a typed literal the value parser cannot read must show up as a mismatch, not silently pass
    tr = sphn_triples(REC, 0) + [(patient_iri(0), '<http://nvasc.org/odd>', '"n/a"^^<http://nvasc.org/dt/age>')]
    info = to_processed(tr, tmp_path / 'g', n_patients=1)
    assert info['literals_expected'] == info['literals_with_value'] + 1


def test_check_graph_ignores_other_patients_values_reached_through_shared_nodes():
    # two patients share a class node; with inverse edges each patient reaches the other's value at 4+ hops
    tr = []
    for i in (0, 1):
        rec = f'<http://nvasc.org/p{i}/age>'
        tr += [(rec, '<pid>', patient_iri(i)), (rec, '<type>', '<http://example.org/Age>'),
               (rec, '<value>', f'"{60 + i}.0"^^<http://nvasc.org/dt/age>')]
    both = tr + [(o, '<inverse>', h) for (h, _, o) in tr]
    assert check_graph(both, layers=2, n_patients=2) == []


def test_check_graph_can_check_a_single_patient_subgraph():
    from build_sphn_real import sphn_real_triples
    tr = sphn_real_triples(REC, 7)
    both = tr + [(o, '<inverse>', h) for (h, _, o) in tr]
    assert check_graph(both, layers=4, patient_ids=[7]) == []
    assert any('beyond' in p for p in check_graph(both, layers=2, patient_ids=[7]))
