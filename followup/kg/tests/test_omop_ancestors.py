"""OMOP ancestor extraction on a tiny synthetic vocabulary (no licensed content)."""
from omop_ancestors import main, iri
from build_sphn_real import SNOMED


def write(path, header, rows):
    path.write_text('\t'.join(header) + '\n' + ''.join('\t'.join(map(str, r)) + '\n' for r in rows))


def test_transitive_is_a_ancestors_without_vocabulary_roots(tmp_path):
    write(tmp_path / 'CONCEPT.csv', ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code'],
          [(1, 'Glasgow coma scale', 'x', 'SNOMED', '386554004'), (2, 'Assessment scales', 'x', 'SNOMED', '273249006'),
           (3, 'Staging and scales', 'x', 'SNOMED', '254291000'), (4, 'SNOMED CT Concept', 'x', 'SNOMED', '138875005'),
           (5, 'Old parent', 'x', 'SNOMED', '999')])
    write(tmp_path / 'CONCEPT_RELATIONSHIP.csv', ['concept_id_1', 'concept_id_2', 'relationship_id', 'invalid_reason'],
          [(1, 2, 'Is a', ''), (2, 3, 'Is a', ''), (3, 4, 'Is a', ''), (1, 5, 'Is a', 'D'), (2, 1, 'Subsumes', '')])
    main(str(tmp_path), tmp_path / 'out.tsv')
    rows = [l.rstrip('\n').split('\t') for l in open(tmp_path / 'out.tsv')]
    assert rows == [[f'<{SNOMED}386554004>', f'<{SNOMED}273249006>', 'Assessment scales', '1'],
                    [f'<{SNOMED}386554004>', f'<{SNOMED}254291000>', 'Staging and scales', '2']]


def test_codes_with_slashes_become_valid_iris():
    assert iri('LOINC', 'HEM/BC') == '<https://loinc.org/rdf/HEM%2FBC>'
