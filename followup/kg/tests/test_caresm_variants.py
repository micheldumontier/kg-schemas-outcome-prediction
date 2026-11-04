"""Tests for the matched CARE-SM variants, on two synthetic records only."""
import joblib
import numpy as np
import pytest
from build_graphs import caresm_triples, check_graph, hop_distances, patient_iri, nv, SIO, TYPE
from caresm_variants import (VARIANTS, base_triples, build, max_hops, command, run_name, LAYERS,
                             HAS_START, HAS_ATTRIBUTE, PARTICIPANT_IN, write_variants)

RECS = [{'age': 61.0, 'gender': 'F', 'gcs_first': 9.0, 'sodium_min': 131.0,
         'nimodipine': 1, 'nimodipine_h': 5.0, 'evd': 1, 'evd_h': 2.0, 'coiling': 0, 'coiling_h': np.nan},
        {'age': 47.0, 'gender': 'M', 'gcs_first': 14.0, 'sodium_min': np.nan,
         'nimodipine': 1, 'nimodipine_h': 1.0, 'evd': 0, 'evd_h': np.nan, 'coiling': 1, 'coiling_h': 3.0}]
ATTRIBUTES = {0: ['gender_F', 'age', 'gcs_first', 'sodium_min'], 1: ['gender_M', 'age', 'gcs_first']}
EVENTS_OF = {0: ['nimodipine', 'evd'], 1: ['nimodipine', 'coiling']}
HAS_VALUE = f'<{SIO}has-value>'


def graphs():
    return {name: build(name, RECS) for name in VARIANTS}


def extra(name, over='caresm_a'):
    g = graphs()
    assert set(g[over]) <= set(g[name])
    return set(g[name]) - set(g[over])


def test_variant_names_and_predicate_iris():
    assert set(VARIANTS) == {'caresm_a', 'caresm_sio', 'caresm_feat', 'caresm_sio_part'}
    assert HAS_START == '<http://semanticscience.org/resource/SIO_000680>'
    assert HAS_ATTRIBUTE == '<http://semanticscience.org/resource/SIO_000008>'
    assert PARTICIPANT_IN == '<http://semanticscience.org/resource/SIO_000062>'


def test_base_links_event_times_with_sio_start_time_and_has_value():
    a = base_triples(RECS[0], 0)
    assert not any(p == nv('hasTimePoint') for (_, p, _) in a)
    assert (nv('p0/nimodipine_process'), HAS_START, nv('p0/nimodipine_start')) in a
    assert (nv('p0/nimodipine_start'), HAS_VALUE, '"5.0"^^<http://nvasc.org/dt/hours>') in a
    assert not any('_context' in x for t in a for x in t)
    # everything else is caresm_triples as it is: same size, and the non-time triples are identical
    c = caresm_triples(RECS[0], 0)
    assert len(a) == len(c)
    time_c = {t for t in c if t[1] in (nv('hasTimePoint'), f'<{SIO}SIO_000680>')}
    time_a = {t for t in a if t[1] == HAS_START or t[0].endswith('_start>')}
    assert set(c) - time_c == set(a) - time_a
    assert len(time_a) == len(time_c) == 4


def test_a_has_no_shortcut_from_the_patient():
    a = build('caresm_a', RECS)
    from_patient = {p for (h, p, _) in a if h in (patient_iri(0), patient_iri(1))}
    assert from_patient == {TYPE, f'<{SIO}has-role>'}
    assert not any(p == nv('hasTimePoint') for (_, p, _) in a)


def test_b_adds_one_shared_has_attribute_edge_per_attribute_output():
    e = extra('caresm_sio')
    expected = {(patient_iri(i), HAS_ATTRIBUTE, nv(f'p{i}/{f.split("_")[0] if f.startswith("gender") else f}_output'))
                for i, fs in ATTRIBUTES.items() for f in fs}
    assert e == expected
    assert {p for (_, p, _) in e} == {HAS_ATTRIBUTE}


def test_c_adds_one_predicate_per_attribute_and_none_for_events_or_times():
    e = extra('caresm_feat')
    expected = {(patient_iri(i), nv(f'compiled/has_{f}'),
                 nv(f'p{i}/{"gender" if f.startswith("gender") else f}_output'))
                for i, fs in ATTRIBUTES.items() for f in fs}
    assert e == expected
    preds = {p for (_, p, _) in e}
    assert not any(ev in p for p in preds for ev in ('nimodipine', 'evd', 'coiling', '_time'))


def test_d_adds_participation_and_process_typing_to_b():
    e = extra('caresm_sio_part', over='caresm_sio')
    expected = set()
    for i, evs in EVENTS_OF.items():
        for ev in evs:
            proc = nv(f'p{i}/{ev}_process')
            expected |= {(patient_iri(i), PARTICIPANT_IN, proc), (proc, TYPE, nv(f'event/{ev}'))}
    assert e == expected


@pytest.mark.parametrize('name,attr,time', [('caresm_a', 4, 4), ('caresm_sio', 2, 4), ('caresm_feat', 2, 4),
                                            ('caresm_sio_part', 2, 3)])
def test_hop_distances_to_attribute_values_and_event_times(name, attr, time):
    assert max_hops(build(name, RECS), patient_ids=[0, 1]) == {'attribute_values': attr, 'event_times': time}


def test_every_variant_passes_the_check_at_four_layers_and_a_fails_at_three():
    for name in VARIANTS:
        assert check_graph(build(name, RECS), layers=4, n_patients=2) == []
    assert any('beyond' in p for p in check_graph(build('caresm_a', RECS), layers=3, n_patients=2))


def test_write_variants_writes_runner_files_and_checks_outcomes(tmp_path):
    reference = tmp_path / 'ref_outcomes.joblib'
    joblib.dump([0, 2], reference)
    rows = write_variants(RECS, [0, 2], tmp_path / 'out', reference)
    assert {r['variant'] for r in rows} == set(VARIANTS)
    assert all(r['check'] == 'ok' for r in rows)
    for name in VARIANTS:
        for suffix in ('_entities.tsv', '_triples.tsv', '_numeric.npy', '_relations.tsv'):
            assert (tmp_path / 'out' / f'{name}{suffix}').exists()
    assert joblib.load(tmp_path / 'out' / 'outcomes.joblib') == [0, 2]
    joblib.dump([0, 1], reference)
    with pytest.raises(ValueError, match='outcomes'):
        write_variants(RECS, [0, 2], tmp_path / 'out2', reference)


def test_run_command_matches_the_mie_caresm_runs():
    assert LAYERS == (3, 4)
    assert run_name(0, 'caresm_sio', 3) == 'MIMIC24_caresm_sio_L3'
    assert run_name(2, 'caresm_a', 4) == 'MIMIC24s2_caresm_a_L4'
    cmd = command(0, 'caresm_feat', 3, n_patients=1234)
    s = ' '.join(cmd)
    assert '--both_directions' not in cmd
    assert cmd[cmd.index('--layers') + 1] == '3'
    assert cmd[cmd.index('--run') + 1] == 'MIMIC24_caresm_feat_L3'
    assert cmd[cmd.index('--num_patients') + 1] == '1234'
    assert cmd[cmd.index('--zero_emb_regex') + 1] == r'synth_patient_|/p[0-9]+/|\^\^'
    assert cmd[cmd.index('--prefix') + 1].endswith('caresm_variants/graphs_L24/caresm_feat')
    assert cmd[cmd.index('--results_dir') + 1].endswith('mimic-iv/derived/caresm_variants/results')
    for flag, value in [('--emb_std', '1.0'), ('--epochs', '2000'), ('--device', 'cuda'), ('--seed', '0')]:
        assert cmd[cmd.index(flag) + 1] == value
    assert 'repro/run_rgcn.py' in s
