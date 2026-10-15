from pathlib import Path
import pytest
from terminology import ancestors, ATC, PCS, SPHN, LOINC, SPHN_PARENTS, PCS_AXES

SCHEMA = Path(__file__).resolve().parents[3] / 'ontologies/sphn/2026-1/rdf_schema/sphn_rdf_schema.ttl'


def test_atc_ancestors_follow_the_code_levels():
    got = ancestors(f'<{ATC}C08CA06>')
    assert [a for a, _ in got] == [f'<{ATC}C08CA>', f'<{ATC}C08C>', f'<{ATC}C08>', f'<{ATC}C>']
    assert got[0][1] == 'dihydropyridine derivatives'
    assert [a for a, _ in ancestors(f'<{ATC}C01CA>')] == [f'<{ATC}C01C>', f'<{ATC}C01>', f'<{ATC}C>']


def test_coiling_and_clipping_share_a_body_part_ancestor_that_drainage_does_not():
    coil = {a for a, _ in ancestors(f'<{PCS}03VG3DZ>')}
    clip = {a for a, _ in ancestors(f'<{PCS}03VG0CZ>')}
    evd = {a for a, _ in ancestors(f'<{PCS}009630Z>')}
    assert f'<{PCS}03VG>' in coil & clip and coil & evd == {f'<{PCS}0>'}
    assert dict(ancestors(f'<{PCS}03VG3DZ>'))[f'<{PCS}03VG>'] == 'Restriction of Intracranial Artery'


def test_sphn_classes_get_their_superclass_without_the_universal_root():
    assert [a for a, _ in ancestors(f'<{SPHN}HeartRate>')] == [f'<{SPHN}Result>']
    assert ancestors(f'<{SPHN}Age>') == []
    assert ancestors(f'<{SPHN}SPHNConcept>') == [] and ancestors(f'<{LOINC}2951-2>') == []


def test_pcs_codes_are_seven_characters_with_seven_axes():
    assert all(len(c) == 7 and len(a) == 7 for c, a in PCS_AXES.items())


@pytest.mark.skipif(not SCHEMA.exists(), reason='SPHN 2026.1 schema not fetched (ontologies/fetch_sphn.sh)')
def test_sphn_parents_match_the_released_schema():
    import rdflib
    from rdflib.namespace import RDFS
    g = rdflib.Graph().parse(SCHEMA)
    for cls, parent in SPHN_PARENTS.items():
        assert rdflib.URIRef(SPHN + parent) in set(g.objects(rdflib.URIRef(SPHN + cls), RDFS.subClassOf)), cls


def test_ancestors_of_an_ancestor_are_its_own_prefixes():
    assert [a for a, _ in ancestors(f'<{PCS}03VG>')] == [f'<{PCS}03V>', f'<{PCS}03>', f'<{PCS}0>']
    assert ancestors(f'<{PCS}0>') == [] and ancestors(f'<{ATC}C>') == []
