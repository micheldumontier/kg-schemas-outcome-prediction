from pathlib import Path
import pytest
import rdflib
from rdflib.namespace import RDFS
from build_ontology import code_classes, assemble
from terminology import ATC, PCS, LOINC

SCHEMA = Path(__file__).resolve().parents[3] / 'ontologies/sphn/2026-1/rdf_schema/sphn_rdf_schema.ttl'


def test_every_code_class_has_a_label_and_parents_are_classes_too():
    cc = code_classes()
    assert cc[ATC + 'C08CA06'] == ('nimodipine', ATC + 'C08CA')
    assert cc[PCS + '03VG3DZ'][1] == PCS + '03VG' and cc[PCS + '03VG'][1] == PCS + '03V' and cc[PCS + '0'][1] is None
    assert cc[LOINC + '2160-0'][0].startswith('Creatinine')
    assert all(label for label, _ in cc.values())
    assert all(parent in cc for _, parent in cc.values() if parent)


@pytest.mark.skipif(not SCHEMA.exists(), reason='SPHN 2026.1 schema not fetched (ontologies/fetch_sphn.sh)')
def test_assembled_ontology_round_trips_as_rdf_xml(tmp_path):
    g = assemble(SCHEMA)
    g.serialize(tmp_path / 'o.owl', format='pretty-xml')
    h = rdflib.Graph().parse(tmp_path / 'o.owl', format='xml')
    assert len(h) == len(g)
    assert (rdflib.URIRef(ATC + 'C08CA06'), RDFS.subClassOf, rdflib.URIRef(ATC + 'C08CA')) in h
    assert not list(h.triples((None, rdflib.namespace.OWL.imports, None)))
