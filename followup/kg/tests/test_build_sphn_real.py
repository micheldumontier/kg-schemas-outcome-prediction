"""Tests for the real-SPHN (2026.1) graph builder, on synthetic records only."""
import numpy as np
from build_sphn_real import sphn_real_triples, shared_labels, SPHN, ATC, PCS, SNOMED
from build_graphs import hop_distances, patient_iri, to_processed

REC = {'age': 61.0, 'gender': 'F', 'gcs_first': 9.0, 'hr_max': 110.0, 'sbp_min': 95.0, 'sodium_min': 131.0,
       'nimodipine': 1, 'nimodipine_h': 5.0, 'evd': 1, 'evd_h': 2.0, 'coiling': 0, 'coiling_h': np.nan}


def test_records_point_to_the_patient_with_real_sphn_terms():
    tr = sphn_real_triples(REC, 0)
    p = patient_iri(0)
    assert (p, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', f'<{SPHN}SubjectPseudoIdentifier>') in tr
    to_patient = [h for (h, r, o) in tr if r == f'<{SPHN}hasSubjectPseudoIdentifier>' and o == p]
    assert len(to_patient) >= 6
    assert all(r.startswith(f'<{SPHN}') or r.endswith(('rdf-syntax-ns#type>', 'rdf-schema#label>')) for (_, r, _) in tr)


def test_codes_use_real_terminologies():
    objs = {o for (_, _, o) in sphn_real_triples(REC, 0)}
    assert f'<{ATC}C08CA06>' in objs           # nimodipine
    assert f'<{PCS}009630Z>' in objs           # ventricular drainage, percutaneous
    assert f'<{SNOMED}248152002>' in objs      # female
    assert f'<{SNOMED}386554004>' in objs      # Glasgow coma scale
    assert not any('coiling' in o or '03VG3DZ' in o for o in objs)


def test_patient_reaches_every_value_within_four_hops_when_edges_go_both_ways():
    tr = sphn_real_triples(REC, 0)
    one_way = hop_distances(tr, patient_iri(0))
    both = hop_distances(tr + [(o, 'inv', h) for (h, _, o) in tr], patient_iri(0))
    lits = {o for (_, _, o) in tr if o.startswith('"') and '^^' in o}
    assert not (lits & set(one_way))                       # patient receives nothing in the native direction
    assert lits <= set(both) and max(both[l] for l in lits) <= 4


def test_shared_nodes_have_human_readable_labels():
    tr = sphn_real_triples(REC, 0)
    labels = shared_labels(tr)
    assert labels[f'<{SPHN}HeartRate>'] == 'Heart Rate'
    assert labels[f'<{ATC}C08CA06>'] == 'nimodipine'
    assert labels[f'<{PCS}009630Z>'].startswith('Drainage of Cerebral Ventricle')
    assert not any('/p0/' in k or 'synth_patient' in k for k in labels)


def test_labs_carry_loinc_codes():
    from terminology import LOINC
    objs = {o for (_, _, o) in sphn_real_triples(REC, 0)}
    assert f'<{LOINC}2951-2>' in objs and not any('nvasc.org/lab/' in o for o in objs)
    assert shared_labels(sphn_real_triples(REC, 0))[f'<{LOINC}2951-2>'].startswith('Sodium [Moles/volume]')


def test_hierarchy_closure_keeps_ancestors_at_the_depth_of_their_code():
    from build_sphn_real import with_hierarchy, SUBCLASS
    base = sphn_real_triples(REC, 0)
    (tr,), onto = with_hierarchy([base])
    both = lambda t: t + [(o, 'inv', h) for (h, _, o) in t]
    d0, d1 = hop_distances(both(base), patient_iri(0)), hop_distances(both(tr), patient_iri(0))
    assert d1[f'<{ATC}C08CA>'] == d0[f'<{ATC}C08CA06>'] and d1[f'<{PCS}0>'] == d0[f'<{PCS}009630Z>']
    assert d1[f'<{SPHN}Result>'] == d0[f'<{SPHN}HeartRate>'] and f'<{SPHN}SPHNConcept>' not in d1
    assert (f'<{ATC}C08CA06>', SUBCLASS, f'<{ATC}C08>') in onto and len(set(onto)) == len(onto)
    labels = shared_labels(tr + onto)
    assert labels[f'<{ATC}C08>'] == 'calcium channel blockers' and labels[f'<{SPHN}Result>'] == 'Result'
    assert labels[f'<{ATC}C08CA06>'] == 'nimodipine'
