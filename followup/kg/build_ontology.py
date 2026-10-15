"""Assembled ontology for the OWL2Vec* arm (plan A, semantic enrichment).

Usage: python build_ontology.py <sphn_rdf_schema.ttl> <out.owl>
Merges the SPHN RDF Schema 2026.1 with OWL classes for every code used in the real-SPHN graphs: ATC and ICD-10-PCS codes
with their ancestors (rdfs:subClassOf the direct parent), LOINC laboratory codes and SNOMED CT concepts, each with its
rdfs:label. The schema's owl:imports are dropped. Writes RDF/XML, which OWL2Vec* (through owlready2) can load.
"""
import sys
import rdflib
from rdflib.namespace import OWL, RDF, RDFS
from terminology import ATC, PCS, LOINC, ATC_LABELS, PCS_AXES, LOINC_LABS, ancestors
from build_sphn_real import SNOMED, SNOMED_LABELS, PROC_CODES


def code_classes():
    """(class IRI, label, direct parent IRI or None) for every code and ancestor used in the graphs."""
    out = {}
    for code, label in ATC_LABELS.items():
        anc = ancestors(f'<{ATC}{code}>')
        out[ATC + code] = (label, anc[0][0][1:-1] if anc else None)
    for code in PCS_AXES:
        title = next(name for c, name in PROC_CODES.values() if c == code)
        chain = [(f'<{PCS}{code}>', title)] + ancestors(f'<{PCS}{code}>')
        for (iri, label), parent in zip(chain, chain[1:] + [None]):
            out.setdefault(iri[1:-1], (label, parent[0][1:-1] if parent else None))
    for code, label in LOINC_LABS.values():
        out[LOINC + code] = (label, None)
    for code, label in SNOMED_LABELS.items():
        out[SNOMED + code] = (label, None)
    return out


def assemble(schema_path):
    g = rdflib.Graph()
    g.parse(schema_path)
    g.remove((None, OWL.imports, None))      # imported ontologies (DICOM and others) are not needed and not fetched
    for iri, (label, parent) in code_classes().items():
        c = rdflib.URIRef(iri)
        g.add((c, RDF.type, OWL.Class))
        g.add((c, RDFS.label, rdflib.Literal(label, lang='en')))
        if parent:
            g.add((c, RDFS.subClassOf, rdflib.URIRef(parent)))
    return g


if __name__ == '__main__':
    g = assemble(sys.argv[1])
    g.serialize(sys.argv[2], format='pretty-xml')
    print(f'wrote {sys.argv[2]} with {len(g)} triples and {len(code_classes())} code classes')
